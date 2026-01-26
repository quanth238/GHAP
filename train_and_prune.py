#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# from gaussian_splatting_sampler.utils import  RobustAutoSamplingTrigger
# from gaussian_splatting_sampler.gmm_sampler import gaussian_model_reduction
from gmm_sampler import gaussian_model_reduction
import json
import time
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from compaction.rss_voxel import RSSVoxelConfig, build_student_from_rss_voxel
from compaction.greedy_coverage import GreedyCoverageConfig, build_student_from_greedy_coverage
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def _log_optimizer_state(optimizer, tag: str) -> None:
    print(f"[RSS][Debug] Optimizer state ({tag})")
    for group in optimizer.param_groups:
        name = group.get("name", "unknown")
        if not group["params"]:
            print(f"[RSS][Debug]  {name}: no params")
            continue
        param = group["params"][0]
        state = optimizer.state.get(param, None)
        if not state or "exp_avg" not in state:
            print(f"[RSS][Debug]  {name}: no exp_avg state")
            continue
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        flat = exp_avg.view(-1)
        step = max(1, flat.numel() // 200000)
        sample = flat[::step]
        mean_abs = float(sample.abs().mean().item())
        max_abs = float(sample.abs().max().item())
        flat2 = exp_avg_sq.view(-1)[::step]
        mean_sq = float(flat2.mean().item())
        print(
            f"[RSS][Debug]  {name}: exp_avg_mean_abs={mean_abs:.6e} "
            f"exp_avg_max_abs={max_abs:.6e} exp_avg_sq_mean={mean_sq:.6e}"
        )

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, compaction=None):
    if compaction.flag:
        print('With Compaction!')
    else:
        print('No Compaction!')

    start_time = time.time()
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only = False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    time_budget_sec = 0.0
    if getattr(opt, "time_budget_minutes", 0.0) and opt.time_budget_minutes > 0:
        time_budget_sec = float(opt.time_budget_minutes) * 60.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        ######################
        # TODO: gaussians.update_learning_rate(iteration, False)
        # if iteration == 30000:
        #     gaussians.update_learning_rate(iteration, True)
        # else:
        #     gaussians.update_learning_rate(iteration, True)
        gaussians.update_learning_rate(iteration, False)
        ######################
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()
        elapsed_time = iter_start.elapsed_time(iter_end)

        with torch.no_grad():
            skip_step = False
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                # TODO: subsampling in training
            if compaction.flag:
                if iteration in compaction.iter:
                    index = compaction.iter.index(iteration)
                    if compaction.method == "rss_voxel":
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        pre_compact_count = int(gaussians.get_xyz.shape[0])
                        if not getattr(compaction, "logged_counts", False):
                            print(f"[RSS] Gaussians pre-compaction: {pre_compact_count}")
                        target_k = compaction.target_num_gaussians
                        if target_k <= 0:
                            ratio = compaction.ratio[index]
                            target_k = int(ratio) if ratio > 1 else max(1, int(gaussians.get_xyz.shape[0] * ratio))
                        target_k = min(target_k, gaussians.get_xyz.shape[0])
                        print(f"[RSS] Target K={target_k}")

                        cfg = RSSVoxelConfig(
                            target_num_gaussians=target_k,
                            num_views=compaction.rss_num_views,
                            pixels_per_view=compaction.rss_pixels_per_view,
                            alpha_tau=compaction.rss_alpha_tau,
                            lambda_tex=compaction.rss_lambda_tex,
                            hit_quantile=compaction.rss_hit_quantile,
                            depth_var_thresh=compaction.rss_depth_var_thresh,
                            center_mode=compaction.rss_center_mode,
                            teacher_selector=compaction.rss_teacher_selector,
                            mass_source=compaction.rss_mass_source,
                            mass_topk=compaction.rss_mass_topk,
                            no_reset=compaction.rss_no_reset,
                            voxel_search=compaction.rss_voxel_search,
                            voxel_search_iters=compaction.rss_voxel_search_iters,
                            voxel_size=compaction.rss_voxel_size,
                            depth_gate=compaction.rss_depth_gate,
                            seed=compaction.rss_seed,
                            debug=compaction.rss_debug,
                            debug_samples=compaction.rss_debug_samples,
                            snap_to_teacher=compaction.rss_snap_to_teacher,
                            snap_factor=compaction.rss_snap_factor,
                            snap_min=compaction.rss_snap_min,
                            snap_unique=compaction.rss_snap_unique,
                            snap_fill_teacher=compaction.rss_snap_fill_teacher,
                        )

                        new_params, timings = build_student_from_rss_voxel(
                            gaussians, scene, dataset, pipe, cfg, SPARSE_ADAM_AVAILABLE
                        )
                        if new_params is None:
                            print("[RSS] Compaction skipped; continuing with teacher.")
                            if not getattr(compaction, "logged_counts", False):
                                post_compact_count = int(gaussians.get_xyz.shape[0])
                                print(f"[RSS] Gaussians post-compaction: {post_compact_count}")
                                compaction.logged_counts = True
                                compaction.did_compact = True
                        else:
                            compaction.voxel_size = timings.get("voxel_size", None)
                            if compaction.rss_debug:
                                _log_optimizer_state(gaussians.optimizer, "pre_compaction")
                            xyz_tensors = gaussians.replace_tensor_to_optimizer(new_params["xyz"], "xyz")
                            gaussians._xyz = xyz_tensors["xyz"]
                            f_dc_tensors = gaussians.replace_tensor_to_optimizer(new_params["f_dc"], "f_dc")
                            gaussians._features_dc = f_dc_tensors["f_dc"]
                            f_rest_tensors = gaussians.replace_tensor_to_optimizer(new_params["f_rest"], "f_rest")
                            gaussians._features_rest = f_rest_tensors["f_rest"]
                            opacity_tensors = gaussians.replace_tensor_to_optimizer(new_params["opacity"], "opacity")
                            gaussians._opacity = opacity_tensors["opacity"]
                            scaling_tensors = gaussians.replace_tensor_to_optimizer(new_params["scaling"], "scaling")
                            gaussians._scaling = scaling_tensors["scaling"]
                            rotation_tensors = gaussians.replace_tensor_to_optimizer(new_params["rotation"], "rotation")
                            gaussians._rotation = rotation_tensors["rotation"]

                            gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
                            gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
                            gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                            if torch.cuda.is_available():
                                peak_mem = torch.cuda.max_memory_allocated()
                                print(f"[RSS] Peak CUDA memory: {peak_mem / (1024 ** 3):.2f} GB")
                            print(
                                "[RSS] Timing: render+sample={:.2f}s voxel={:.2f}s "
                                "kdtree={:.2f}s total={:.2f}s voxel_size={:.6f} "
                                "M={:.0f} K={:.0f}".format(
                                    timings.get("render_sampling", 0.0),
                                    timings.get("voxel", 0.0),
                                    timings.get("kdtree", 0.0),
                                    timings.get("total", 0.0),
                                    timings.get("voxel_size", 0.0),
                                    timings.get("num_samples", 0.0),
                                    timings.get("num_centers", 0.0),
                                )
                            )
                            if "scale_low_ratio" in timings and not getattr(compaction, "logged_clamp_ratios", False):
                                print(
                                    "[RSS] Clamp ratios: scale_low={:.2f}% scale_high={:.2f}% "
                                    "opacity_low={:.2f}% opacity_high={:.2f}%".format(
                                        100.0 * timings.get("scale_low_ratio", 0.0),
                                        100.0 * timings.get("scale_high_ratio", 0.0),
                                        100.0 * timings.get("opacity_low_ratio", 0.0),
                                        100.0 * timings.get("opacity_high_ratio", 0.0),
                                    )
                                )
                                compaction.logged_clamp_ratios = True
                            if not getattr(compaction, "logged_counts", False):
                                post_compact_count = int(gaussians.get_xyz.shape[0])
                                print(f"[RSS] Gaussians post-compaction: {post_compact_count}")
                                compaction.logged_counts = True
                                compaction.did_compact = True
                            compaction.finetune_start = time.time()
                            skip_step = True
                            if compaction.rss_debug:
                                _log_optimizer_state(gaussians.optimizer, "post_compaction")
                    elif compaction.method == "greedy_coverage":
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        pre_compact_count = int(gaussians.get_xyz.shape[0])
                        if not getattr(compaction, "logged_counts", False):
                            print(f"[GC] Gaussians pre-compaction: {pre_compact_count}")
                        target_k = compaction.target_num_gaussians
                        if target_k <= 0:
                            ratio = compaction.ratio[index]
                            target_k = int(ratio) if ratio > 1 else max(1, int(gaussians.get_xyz.shape[0] * ratio))
                        target_k = min(target_k, gaussians.get_xyz.shape[0])
                        print(f"[GC] Target K={target_k}")

                        cfg = GreedyCoverageConfig(
                            target_num_gaussians=target_k,
                            num_views=compaction.gc_num_views,
                            pixels_per_view=compaction.gc_pixels_per_view,
                            alpha_tau=compaction.gc_alpha_tau,
                            topk_contrib=compaction.gc_topk_contrib,
                            seed=compaction.gc_seed,
                            lazy=compaction.gc_lazy,
                            debug=compaction.gc_debug,
                            log_curve=compaction.gc_log_curve,
                        )

                        new_params, timings = build_student_from_greedy_coverage(
                            gaussians, scene, dataset, pipe, cfg, SPARSE_ADAM_AVAILABLE
                        )
                        xyz_tensors = gaussians.replace_tensor_to_optimizer(new_params["xyz"], "xyz")
                        gaussians._xyz = xyz_tensors["xyz"]
                        f_dc_tensors = gaussians.replace_tensor_to_optimizer(new_params["f_dc"], "f_dc")
                        gaussians._features_dc = f_dc_tensors["f_dc"]
                        f_rest_tensors = gaussians.replace_tensor_to_optimizer(new_params["f_rest"], "f_rest")
                        gaussians._features_rest = f_rest_tensors["f_rest"]
                        opacity_tensors = gaussians.replace_tensor_to_optimizer(new_params["opacity"], "opacity")
                        gaussians._opacity = opacity_tensors["opacity"]
                        scaling_tensors = gaussians.replace_tensor_to_optimizer(new_params["scaling"], "scaling")
                        gaussians._scaling = scaling_tensors["scaling"]
                        rotation_tensors = gaussians.replace_tensor_to_optimizer(new_params["rotation"], "rotation")
                        gaussians._rotation = rotation_tensors["rotation"]

                        gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
                        gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
                        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            peak_mem = torch.cuda.max_memory_allocated()
                            print(f"[GC] Peak CUDA memory: {peak_mem / (1024 ** 3):.2f} GB")
                        print(
                            "[GC] Timing: render+sample={:.2f}s greedy={:.2f}s total={:.2f}s "
                            "rays={:.0f} pairs={:.0f} K={:.0f}".format(
                                timings.get("render_sampling", 0.0),
                                timings.get("greedy", 0.0),
                                timings.get("total", 0.0),
                                timings.get("num_rays", 0.0),
                                timings.get("num_pairs", 0.0),
                                timings.get("selected", 0.0),
                            )
                        )
                        if "log_path" in timings:
                            print(f"[GC] Curve saved: {timings['log_path']}")
                        if not getattr(compaction, "logged_counts", False):
                            post_compact_count = int(gaussians.get_xyz.shape[0])
                            print(f"[GC] Gaussians post-compaction: {post_compact_count}")
                            compaction.logged_counts = True
                            compaction.did_compact = True
                        compaction.finetune_start = time.time()
                        skip_step = True
                    else:
                        gaussians = subsampling(gaussians, compaction.ratio[index], 42, compaction.method)
            if time_budget_sec > 0 and (time.time() - start_time) >= time_budget_sec:
                if iteration not in saving_iterations:
                    print("\n[TimeBudget] Saving Gaussians at iter {}".format(iteration))
                    scene.save(iteration)
                print(
                    "[TimeBudget] Reached {:.2f} minutes at iter {}. Stopping.".format(
                        time_budget_sec / 60.0, iteration
                    )
                )
                progress_bar.close()
                break
            # Optimizer step
            if iteration < opt.iterations:
                if skip_step:
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if compaction is not None and compaction.method == "rss_voxel":
        if compaction.finetune_start is not None:
            finetune_time = time.time() - compaction.finetune_start
            print(f"[RSS] Finetune time: {finetune_time:.2f}s")
            if compaction.voxel_size and not getattr(compaction, "logged_post_finetune_clamp", False):
                min_scale = max(float(compaction.voxel_size) * 0.1, 1e-4)
                max_scale = max(float(compaction.voxel_size) * 4.0, min_scale * 2.0)
                with torch.no_grad():
                    scales = scene.gaussians.get_scaling.detach()
                    opacities = scene.gaussians.get_opacity.detach()
                    scale_low = float((scales < min_scale).float().mean().item())
                    scale_high = float((scales > max_scale).float().mean().item())
                    op_low = float((opacities < 0.05).float().mean().item())
                    op_high = float((opacities > 0.9).float().mean().item())
                print(
                    "[RSS] Clamp ratios (post-finetune): scale_low={:.2f}% scale_high={:.2f}% "
                    "opacity_low={:.2f}% opacity_high={:.2f}%".format(
                        100.0 * scale_low,
                        100.0 * scale_high,
                        100.0 * op_low,
                        100.0 * op_high,
                    )
                )
                compaction.logged_post_finetune_clamp = True
        if getattr(compaction, "did_compact", False):
            final_count = int(scene.gaussians.get_xyz.shape[0])
            print(f"[RSS] Final gaussians: {final_count}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('gaussians_num', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    # if tb_writer and (idx < 5):
                    if tb_writer:
                    # TODO: CHANGE
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# TODO: For subsampling in training
def subsampling(gaussians, ratio, random_seed=42, method='GMR'):
    if method == 'GMR':
        gaussian_ = gaussian_model_reduction(gaussians, ratio, random_seed)
        xyz_tensors = gaussians.replace_tensor_to_optimizer(gaussian_._xyz, "xyz")
        gaussians._xyz = xyz_tensors["xyz"]
        f_dc_tensors = gaussians.replace_tensor_to_optimizer(gaussian_._features_dc, "f_dc")
        gaussians._features_dc = f_dc_tensors["f_dc"]
        f_rest_tensors = gaussians.replace_tensor_to_optimizer(gaussian_._features_rest, "f_rest")
        gaussians._features_rest = f_rest_tensors["f_rest"]
        scaling_tensors = gaussians.replace_tensor_to_optimizer(gaussian_._scaling, "scaling")
        gaussians._scaling = scaling_tensors["scaling"]
        rotation_tensors = gaussians.replace_tensor_to_optimizer(gaussian_._rotation, "rotation")
        gaussians._rotation = rotation_tensors["rotation"]
        opacity_tensors = gaussians.replace_tensor_to_optimizer(gaussian_._opacity, "opacity")
        gaussians._opacity = opacity_tensors["opacity"]

        gaussians.xyz_gradient_accum = torch.zeros((gaussian_.get_xyz.shape[0], 1), device="cuda")
        gaussians.denom = torch.zeros((gaussian_.get_xyz.shape[0], 1), device="cuda")
        gaussians.max_radii2D = torch.zeros((gaussian_.get_xyz.shape[0]), device="cuda")
    elif method == 'random':
        from torch import nn
        import numpy as np
        pass # TODO: write random part.
        n = gaussians.get_xyz.shape[0]
        downsample_num = int(n * ratio)
        n_total = gaussians.get_xyz.shape[0]
        assert downsample_num <= n_total
        np.random.seed(random_seed)
        keep_indices = np.random.choice(n_total, size=downsample_num, replace=False)
        keep_indices = torch.from_numpy(keep_indices).to("cuda")
        with torch.no_grad():
            new_xyz = gaussians._xyz[keep_indices]
            new_features_dc = gaussians._features_dc[keep_indices]
            new_features_rest = gaussians._features_rest[keep_indices]
            new_scaling = gaussians._scaling[keep_indices]
            new_rotation = gaussians._rotation[keep_indices]
            new_opacity = gaussians._opacity[keep_indices]
            xyz_tensors = gaussians.replace_tensor_to_optimizer(new_xyz, "xyz")
            gaussians._xyz = xyz_tensors["xyz"]
            f_dc_tensors = gaussians.replace_tensor_to_optimizer(new_features_dc, "f_dc")
            gaussians._features_dc = f_dc_tensors["f_dc"]
            f_rest_tensors = gaussians.replace_tensor_to_optimizer(new_features_rest, "f_rest")
            gaussians._features_rest = f_rest_tensors["f_rest"]
            scaling_tensors = gaussians.replace_tensor_to_optimizer(new_scaling, "scaling")
            gaussians._scaling = scaling_tensors["scaling"]
            rotation_tensors = gaussians.replace_tensor_to_optimizer(new_rotation, "rotation")
            gaussians._rotation = rotation_tensors["rotation"]
            opacity_tensors = gaussians.replace_tensor_to_optimizer(new_opacity, "opacity")
            gaussians._opacity = opacity_tensors["opacity"]
            gaussians.xyz_gradient_accum = torch.zeros((new_xyz.shape[0], 1), device="cuda")
            gaussians.denom = torch.zeros((new_xyz.shape[0], 1), device="cuda")
            gaussians.max_radii2D = torch.zeros((new_xyz.shape[0]), device="cuda")
    return gaussians


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--sampling_iter", type=int, nargs='+', default=[15000])
    parser.add_argument("--sampling_ratio", type=float, nargs='+', default=[0.05])
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument("--block_num", type=int, default=3000)
    parser.add_argument("--compaction_method", type=str, default="ghap",
                        choices=["ghap", "rss_voxel", "render_surface_resample", "greedy_coverage"])
    parser.add_argument("--target_num_gaussians", type=int, default=0)
    parser.add_argument("--rss_num_views", type=int, default=300)
    parser.add_argument("--rss_pixels_per_view", type=int, default=100_000)
    parser.add_argument("--rss_alpha_tau", type=float, default=0.05)
    parser.add_argument("--rss_lambda_tex", type=float, default=0.5)
    parser.add_argument("--rss_hit_quantile", type=float, default=0.7)
    parser.add_argument("--rss_depth_var_thresh", type=float, default=0.01)
    parser.add_argument("--rss_center_mode", type=str, default="mean", choices=["mean", "representative", "teacher"])
    parser.add_argument("--rss_teacher_selector", type=str, default="voxel", choices=["voxel", "octree", "topk"])
    parser.add_argument("--rss_mass_source", type=str, default="dominant", choices=["dominant", "opacity"])
    parser.add_argument("--rss_mass_topk", type=int, default=1)
    parser.add_argument("--rss_no_reset", action="store_true", default=False)
    parser.add_argument("--rss_debug", action="store_true", default=False)
    parser.add_argument("--rss_debug_samples", type=int, default=10000)
    parser.add_argument("--rss_snap_to_teacher", action="store_true", default=False)
    parser.add_argument("--rss_snap_factor", type=float, default=5.0)
    parser.add_argument("--rss_snap_min", type=float, default=0.0)
    parser.add_argument("--rss_snap_unique", action="store_true", default=False)
    parser.add_argument("--rss_snap_no_fill_teacher", action="store_true", default=False)
    parser.add_argument("--rss_voxel_search_iters", type=int, default=8)
    parser.add_argument("--rss_voxel_size", type=float, default=0.0)
    parser.add_argument("--rss_no_voxel_search", action="store_true", default=False)
    parser.add_argument("--rss_no_depth_gate", action="store_true", default=False)
    parser.add_argument("--rss_seed", type=int, default=42)
    parser.add_argument("--gc_num_views", type=int, default=200)
    parser.add_argument("--gc_pixels_per_view", type=int, default=50000)
    parser.add_argument("--gc_alpha_tau", type=float, default=0.02)
    parser.add_argument("--gc_topk_contrib", type=int, default=4)
    parser.add_argument("--gc_lazy", action="store_true", default=True)
    parser.add_argument("--gc_no_lazy", action="store_false", dest="gc_lazy")
    parser.add_argument("--gc_seed", type=int, default=42)
    parser.add_argument("--gc_debug", action="store_true", default=False)
    parser.add_argument("--gc_log_curve", action="store_true", default=True)
    parser.add_argument("--gc_no_log_curve", action="store_false", dest="gc_log_curve")
    parser.add_argument("--iteration", type=int, default=None)

    args = parser.parse_args(sys.argv[1:])
    if args.iteration is not None:
        args.iterations = args.iteration
    if args.compaction_method != "ghap":
        args.compact = True
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    class Compact:
        def __init__(
            self,
            compact,
            sampling_iter,
            sampling_ratio,
            random,
            compaction_method,
            target_num_gaussians,
            rss_num_views,
            rss_pixels_per_view,
            rss_alpha_tau,
            rss_lambda_tex,
            rss_hit_quantile,
            rss_depth_var_thresh,
            rss_center_mode,
            rss_teacher_selector,
            rss_mass_source,
            rss_mass_topk,
            rss_no_reset,
            rss_debug,
            rss_debug_samples,
            rss_snap_to_teacher,
            rss_snap_factor,
            rss_snap_min,
            rss_snap_unique,
            rss_snap_no_fill_teacher,
            rss_voxel_search_iters,
            rss_voxel_size,
            rss_no_voxel_search,
            rss_no_depth_gate,
            rss_seed,
            gc_num_views,
            gc_pixels_per_view,
            gc_alpha_tau,
            gc_topk_contrib,
            gc_lazy,
            gc_seed,
            gc_debug,
            gc_log_curve,
        ):
            self.flag = compact
            self.iter = sampling_iter
            self.ratio = sampling_ratio
            if compaction_method == "render_surface_resample":
                compaction_method = "rss_voxel"
            if compaction_method == "rss_voxel":
                self.method = "rss_voxel"
            elif compaction_method == "greedy_coverage":
                self.method = "greedy_coverage"
            else:
                self.method = "random" if random else "GMR"
            self.target_num_gaussians = target_num_gaussians
            self.rss_num_views = rss_num_views
            self.rss_pixels_per_view = rss_pixels_per_view
            self.rss_alpha_tau = rss_alpha_tau
            self.rss_lambda_tex = rss_lambda_tex
            self.rss_hit_quantile = rss_hit_quantile
            self.rss_depth_var_thresh = rss_depth_var_thresh
            self.rss_center_mode = rss_center_mode
            self.rss_teacher_selector = rss_teacher_selector
            self.rss_mass_source = rss_mass_source
            self.rss_mass_topk = rss_mass_topk
            self.rss_no_reset = rss_no_reset
            self.rss_debug = rss_debug
            self.rss_debug_samples = rss_debug_samples
            self.rss_snap_to_teacher = rss_snap_to_teacher
            self.rss_snap_factor = rss_snap_factor
            self.rss_snap_min = rss_snap_min
            self.rss_snap_unique = rss_snap_unique
            self.rss_snap_fill_teacher = not rss_snap_no_fill_teacher
            self.rss_voxel_search_iters = rss_voxel_search_iters
            self.rss_voxel_size = rss_voxel_size
            self.rss_voxel_search = not rss_no_voxel_search
            self.rss_depth_gate = not rss_no_depth_gate
            self.rss_seed = rss_seed
            self.gc_num_views = gc_num_views
            self.gc_pixels_per_view = gc_pixels_per_view
            self.gc_alpha_tau = gc_alpha_tau
            self.gc_topk_contrib = gc_topk_contrib
            self.gc_lazy = gc_lazy
            self.gc_seed = gc_seed
            self.gc_debug = gc_debug
            self.gc_log_curve = gc_log_curve
            self.finetune_start = None
    # Start GUI server, configure and run training
    compaction = Compact(
        args.compact,
        args.sampling_iter,
        args.sampling_ratio,
        args.random,
        args.compaction_method,
        args.target_num_gaussians,
        args.rss_num_views,
        args.rss_pixels_per_view,
        args.rss_alpha_tau,
        args.rss_lambda_tex,
        args.rss_hit_quantile,
        args.rss_depth_var_thresh,
        args.rss_center_mode,
        args.rss_teacher_selector,
        args.rss_mass_source,
        args.rss_mass_topk,
        args.rss_no_reset,
        args.rss_debug,
        args.rss_debug_samples,
        args.rss_snap_to_teacher,
        args.rss_snap_factor,
        args.rss_snap_min,
        args.rss_snap_unique,
        args.rss_snap_no_fill_teacher,
        args.rss_voxel_search_iters,
        args.rss_voxel_size,
        args.rss_no_voxel_search,
        args.rss_no_depth_gate,
        args.rss_seed,
        args.gc_num_views,
        args.gc_pixels_per_view,
        args.gc_alpha_tau,
        args.gc_topk_contrib,
        args.gc_lazy,
        args.gc_seed,
        args.gc_debug,
        args.gc_log_curve,
    )
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, compaction)

    # All done
    print("\nTraining complete.")
