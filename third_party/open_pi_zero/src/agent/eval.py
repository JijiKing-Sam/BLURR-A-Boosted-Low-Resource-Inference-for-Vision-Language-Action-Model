"""
Main eval agent. Only for Simpler for now.

"""

import logging
import os
import time

import hydra
try:  # optional, only needed when录视频
    import imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None
import numpy as np
import simpler_env
import torch

from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time

log = logging.getLogger(__name__)


class EvalAgent:
    def __init__(self, cfg):
        log.info("EvalAgent.__init__ start, task=%s", getattr(cfg.env, "task", None))
        self.n_eval_episode = cfg.n_eval_episode
        self.n_video = cfg.n_video
        self.log_dir = cfg.log_dir
        self.video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        # model
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        use_bf16 = bool(cfg.get("use_bf16", False))
        use_fp16 = bool(cfg.get("use_fp16", False))
        if use_bf16 and use_fp16:
            raise ValueError("Specify at most one of `use_bf16` or `use_fp16`.")
        if use_fp16:
            self.dtype = torch.float16
        elif use_bf16:
            try:
                is_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            except Exception:
                is_supported = False
            if not is_supported:
                log.warning(
                    "BF16 requested but not supported on this GPU; falling back to FP16."
                )
                self.dtype = torch.float16
            else:
                self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        # global backend toggles for speed on H100
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        log.info("Initializing PiZeroInference (this may take some time)...")
        init_t0 = time.time()
        self.model = PiZeroInference(cfg, use_ddp=False)
        log.info("PiZeroInference initialized in %.2f s", time.time() - init_t0)
        self.load_checkpoint(cfg.checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        # enable int8 action quant when requested in cfg
        try:
            self.model.enable_action_quantization()
        except Exception:
            pass
        if cfg.get(
            "use_torch_compile", True
        ):  # model being compiled in the first batch which takes some time
            self.model = torch.compile(self.model, mode="reduce-overhead")
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        self.model.eval()
        log.info(f"Using cuda device: {self.device} dtype: {self.dtype}")
        log_allocated_gpu_memory(log, "loading model")
        self.act_steps = (
            cfg.act_steps
        )  # e.g., run first two steps of predicted four steps
        self.use_prefix_kv_cache = cfg.get("use_prefix_kv_cache", True)

        # env --- no parallelized
        log.info("Creating SimplerEnv env with task='%s'...", cfg.env.task)
        env_t0 = time.time()
        self.env = simpler_env.make(cfg.env.task)
        log.info("SimplerEnv env created in %.2f s", time.time() - env_t0)

        # env specifics
        log.info("Instantiating env adapter: %s", cfg.env.adapter._target_)
        self.env_adapter = hydra.utils.instantiate(cfg.env.adapter)
        log.info("Env adapter instantiated.")

    def run(self):
        """
        Roughly following simpler_env/simple_inference_visual_matching_prepackaged_envs.py

        Assume no obs history for now
        """
        log.info(
            "EvalAgent.run start: n_eval_episode=%d, n_video=%d, act_steps=%d",
            self.n_eval_episode,
            self.n_video,
            self.act_steps,
        )
        env = self.env
        env_adapter = self.env_adapter
        cnt_episode = 0
        successes = []

        # run episodes --- not dealing with subtasks
        env_reset_options = {}
        env_reset_options["obj_init_options"] = {
            "episode_id": cnt_episode,  # this determines the obj inits in bridge
        }
        log.info("Calling env.reset for episode %d ...", cnt_episode)
        reset_t0 = time.time()
        obs, reset_info = env.reset(options=env_reset_options)
        log.info(
            "env.reset for episode %d finished in %.2f s",
            cnt_episode,
            time.time() - reset_t0,
        )
        env_adapter.reset()
        # obs keys: 'agent', 'extra', 'camera_param', 'image'
        # agent: 'qpos', 'qvel', 'eef_pos', 'controller', 'base_pose'
        instruction = env.get_language_instruction()
        recording = self.n_video > 0
        if recording:
            if imageio is None:
                raise ImportError(
                    "需要录制视频(n_video>0)，但未安装 imageio。请先安装: \n"
                    "  conda install -y -c conda-forge imageio imageio-ffmpeg  \n"
                    "或 pip install imageio imageio-ffmpeg"
                )
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免分词器fork警告

            def video_parent_path(x):
                return os.path.join(self.video_dir, f"video_{x}")

            video_writer = imageio.get_writer(video_parent_path(cnt_episode) + ".mp4")
        # is_final_subtask = env.is_final_subtask()
        log.info(
            "Reset info: %s Instruction: %s Max episode length: %s",
            reset_info,
            instruction,
            getattr(env.spec, "max_episode_steps", None),
        )
        # Bridge: {'scene_name': 'bridge_table_1_v1', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': '', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png', 'rgb_overlay_cameras': ['3rd_view_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'episode_model_ids': ['bridge_carrot_generated_modified', 'bridge_plate_objaverse_larger'], 'episode_model_scales': [1.0, 1.0], 'episode_source_obj_name': 'bridge_carrot_generated_modified', 'episode_target_obj_name': 'bridge_plate_objaverse_larger', 'episode_source_obj_init_pose_wrt_robot_base': Pose([0.381995, 0.104536, 0.0175282], [-0.706719, 0.0305475, -0.0305745, -0.706173]), 'episode_target_obj_init_pose_wrt_robot_base': Pose([0.232, -0.047, -0.000468373], [2.00041e-10, -5.10387e-07, -1.6915e-06, -1]), 'episode_id': 5}
        # Fractal: {'scene_name': 'google_pick_coke_can_1_v4', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': 'recolor_tabletop_visual_matching_1', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png', 'rgb_overlay_cameras': ['overhead_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'model_id': 'opened_coke_can', 'model_scale': 1.0, 'distractor_model_ids': None, 'distractor_model_scales': None, 'obj_init_pose_wrt_robot_base': Pose([0.587925, -0.0238302, 0.840576], [0.707052, -0.0081018, -0.01162, -0.70702]), 'orientation': 'laid_vertically'} Instruction: pick coke can Max episode length: 80
        step_in_episode = 0
        while 1:
            # infer action chunk
            log.debug(
                "Episode %d, step %d: calling env_adapter.preprocess...",
                cnt_episode,
                step_in_episode,
            )
            inputs = self.env_adapter.preprocess(env, obs, instruction)
            log.debug(
                "Episode %d, step %d: preprocess done.",
                cnt_episode,
                step_in_episode,
            )
            causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
                self.model.build_causal_mask_and_position_ids(
                    inputs["attention_mask"], dtype=self.dtype
                )
            )
            image_text_proprio_mask, action_mask = (
                self.model.split_full_mask_into_submasks(causal_mask)
            )
            if self.use_prefix_kv_cache:
                model_inputs = {
                    "input_ids": inputs["input_ids"],
                    "pixel_values": inputs["pixel_values"].to(self.dtype),
                    "image_text_proprio_mask": image_text_proprio_mask,
                    "action_mask": action_mask,
                    "vlm_position_ids": vlm_position_ids,
                    "proprio_position_ids": proprio_position_ids,
                    "action_position_ids": action_position_ids,
                    "proprios": inputs["proprios"].to(self.dtype),
                }
            else:
                # No prefix KV caching ablation: re-run VLM+proprio in every flow step.
                model_inputs = {
                    "input_ids": inputs["input_ids"],
                    "pixel_values": inputs["pixel_values"].to(self.dtype),
                    "causal_mask": causal_mask,
                    "vlm_position_ids": vlm_position_ids,
                    "proprio_position_ids": proprio_position_ids,
                    "action_position_ids": action_position_ids,
                    "proprios": inputs["proprios"].to(self.dtype),
                }

            # move to device; favor non_blocking and channels_last for images
            tensors = {}
            for k, v in model_inputs.items():
                if k == "pixel_values" and v.ndim == 4:
                    v = v.contiguous(memory_format=torch.channels_last)
                tensors[k] = v.to(self.device, non_blocking=True)
            model_inputs = tensors
            # using bf16 shows ~0.001 difference in action inferred when using vs. not using kv cache (infer_action_naive, needs to pass in full causal_mask instead), if starting from the same initial noise. no difference when using float32
            # https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
            with torch.inference_mode():
                infer_t0 = time.time()
                if self.use_prefix_kv_cache:
                    actions = self.model(**model_inputs)
                else:
                    infer_action_naive = getattr(self.model, "infer_action_naive", None)
                    if infer_action_naive is None and hasattr(self.model, "_orig_mod"):
                        infer_action_naive = getattr(
                            self.model._orig_mod, "infer_action_naive", None
                        )
                    if infer_action_naive is None:
                        raise AttributeError(
                            "infer_action_naive not found on model; disable torch.compile or "
                            "use prefix KV cache path."
                        )
                    actions = infer_action_naive(**model_inputs)
                log.debug(
                    "Episode %d, step %d: model forward done in %.3f s.",
                    cnt_episode,
                    step_in_episode,
                    time.time() - infer_t0,
                )
                # actions_naive = self.model.infer_action_naive(**inputs_naive)
                # print(torch.allclose(actions, actions_naive))
            env_actions = self.env_adapter.postprocess(actions[0].float().cpu().numpy())

            # environment step
            for env_action in env_actions[: self.act_steps]:
                step_in_episode += 1
                if step_in_episode % 10 == 0:
                    log.info(
                        "Episode %d, env step %d: stepping env...",
                        cnt_episode,
                        step_in_episode,
                    )
                obs, reward, success, truncated, info = env.step(env_action)
                if truncated:
                    break

            # video
            if recording:
                video_writer.append_data(self.env_adapter.get_video_frame(env, obs))

            # update instruction, e.g., pick apple ---> put in top drawer
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction

            # original octo eval only done when timeout, i.e., not upon success
            if truncated:
                successes.append(success)
                log.info(
                    "Episode %d finished. success=%s, total_steps=%d",
                    cnt_episode,
                    success,
                    step_in_episode,
                )
                if recording:
                    video_writer.close()
                    if success:  # rename video with success
                        os.rename(
                            video_parent_path(cnt_episode) + ".mp4",
                            video_parent_path(cnt_episode) + "_success.mp4",
                        )
                cnt_episode += 1

                # quit
                if cnt_episode >= self.n_eval_episode:
                    break

                # reset
                env_reset_options["obj_init_options"] = {
                    "episode_id": cnt_episode,
                }
                obs, reset_info = env.reset(options=env_reset_options)
                env_adapter.reset()
                instruction = env.get_language_instruction()
                log.info(
                    f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
                )
                recording = self.n_video > cnt_episode
                if recording:
                    video_writer = imageio.get_writer(
                        video_parent_path(cnt_episode) + ".mp4"
                    )

        # summary
        success_rate = np.mean(successes)
        log.info("============ Evaluation Summary ============")
        log.info(f"Number of episodes: {cnt_episode}")
        log.info(f"Success rate: {success_rate}")
        log.info("============================================")

    @log_execution_time(log)
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")
