"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import openai
import requests
import torch
from PIL import Image
from sacsperiments import Experiment, Metrics

from agent import (
    PromptAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router, image_utils
from llms.token_usage import TokenUsage


@dataclass
class TaskMetrics(Metrics):
    success: bool = False
    site: str = ""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    num_steps: int = 0
    latency_seconds: float = 0.0
    task_id: int = 0
    # Cumulative metrics across all runs
    cumulative_tasks: int = 0
    cumulative_successes: int = 0
    cumulative_success_rate: float = 0.0
    cumulative_total_tokens: int = 0
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    cumulative_cache_read_tokens: int = 0
    cumulative_cache_write_tokens: int = 0
    cumulative_num_steps: int = 0
    cumulative_latency_seconds: float = 0.0


CUMULATIVE_FIELDS = [
    "cumulative_tasks", "cumulative_successes", "cumulative_success_rate",
    "cumulative_total_tokens", "cumulative_input_tokens", "cumulative_output_tokens",
    "cumulative_cache_read_tokens", "cumulative_cache_write_tokens",
    "cumulative_num_steps", "cumulative_latency_seconds",
]

EMPTY_CUMULATIVE = {f: 0.0 if "rate" in f or "latency" in f else 0 for f in CUMULATIVE_FIELDS}


def _load_cumulative_from_wandb(run_id: str, project: str, entity: str | None = None) -> dict:
    """Load cumulative metrics from the last logged step in the wandb run."""
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    try:
        run = api.run(path)
    except Exception as e:
        logger.warning(f"Could not fetch wandb run {path}: {e}")
        return {}

    # Get only the last row with the cumulative fields
    history = run.history(keys=CUMULATIVE_FIELDS)
    if history.empty:
        return {}

    last_row = history.iloc[-1]
    # Check that cumulative data is actually present
    if last_row.get("cumulative_tasks", 0) == 0:
        return {}

    return {f: last_row.get(f, EMPTY_CUMULATIVE[f]) for f in CUMULATIVE_FIELDS}


DATASET = os.environ["DATASET"]

LOG_FOLDER = "/mnt/nfs/home/abuzakuk/vwa/log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

# logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument("--render", action="store_true", help="Render the browser")

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)

    # experiment tracking
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--reset_run_id",
        action="store_true",
        help="Reset the saved wandb run ID and start a new run",
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if args.action_set_tag == "id_accessibility_tree" and args.observation_type not in [
        "accessibility_tree",
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [action["action_type"] == ActionTypes.NONE for action in last_k_actions]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all([is_equivalent(action, last_action) for action in last_k_actions]):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
            return True, f"Same typing action for {k} times"

    return False, ""


def test(args: argparse.Namespace, config_file_list: list[str]) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if DATASET == "visualwebarena":
        if caption_image_fn and args.eval_captioning_model == args.captioning_model:
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                torch.float16
                if (
                    torch.cuda.is_available()
                    and args.eval_captioning_model_device == "cuda"
                )
                else torch.float32,
                args.eval_captioning_model,
            )
    else:
        caption_image_fn = None
        eval_caption_image_fn = None

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    def _get_run_id_file() -> Path:
        """Get the path to the file storing the wandb run ID for this experiment.

        Uses a hash of experiment name, wandb project, and entity so that
        different experiments can share the same log folder without collisions.
        """
        import hashlib

        key = f"{args.experiment_name}|{args.wandb_project}|{args.wandb_entity}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        return Path(LOG_FOLDER) / f"wandb_run_id_{digest}.txt"

    def _run_tasks(exp: Experiment | None) -> None:
        if exp:
            run_id = args.run_id
            run_id_file = _get_run_id_file() if args.experiment_name else None

            # If --reset_run_id, delete the saved file
            if args.reset_run_id and run_id_file and run_id_file.exists():
                run_id_file.unlink()
                logger.info(f"Reset wandb run ID (deleted {run_id_file})")

            # If no run_id provided, try to load from saved file
            if not run_id and run_id_file and run_id_file.exists():
                run_id = run_id_file.read_text().strip()
                logger.info(f"Resuming wandb run {run_id} from {run_id_file}")

            if run_id:
                wandb_run = exp.init_wandb_run(
                    run_id=run_id, resume="allow", tags=[args.model, "visualwebarena"]
                )
            else:
                wandb_run = exp.init_wandb_run(tags=[args.model, "visualwebarena"])

            # Save the run ID for future invocations
            if run_id_file:
                run_id_file.write_text(wandb_run.id)
                logger.info(f"Saved wandb run ID {wandb_run.id} to {run_id_file}")

            exp.define_metrics(TaskMetrics)

        # Load cumulative metrics from the last wandb step
        cumulative = dict(EMPTY_CUMULATIVE)
        if exp:
            actual_run_id = wandb_run.id if wandb_run else None
            if actual_run_id and args.wandb_project:
                loaded = _load_cumulative_from_wandb(
                    actual_run_id, args.wandb_project, args.wandb_entity
                )
                if loaded:
                    cumulative = loaded
                    logger.info(
                        f"Loaded cumulative metrics from wandb: "
                        f"{cumulative['cumulative_tasks']} tasks, "
                        f"{cumulative['cumulative_successes']} successes"
                    )

        for _, config_file in enumerate(config_file_list):
            task_usage = TokenUsage()
            task_start_time = time.time()
            score = 0.0
            task_site = ""
            trajectory: Trajectory = []
            task_id = 0

            try:
                print("CREATING RENDER")
                render_helper = RenderHelper(
                    config_file, args.result_dir, args.action_set_tag
                )
                print("DONE")

                # Load task.
                with open(config_file) as f:
                    _c = json.load(f)
                    intent = _c["intent"]
                    task_id = _c["task_id"]
                    image_paths = _c.get("image", None)
                    images = []

                    # automatically login
                    if _c["storage_state"]:
                        cookie_file_name = os.path.basename(_c["storage_state"])
                        comb = get_site_comb_from_filepath(cookie_file_name)
                        task_site = ".".join(comb)
                        temp_dir = tempfile.mkdtemp()

                        print("ARGS")
                        print(temp_dir)
                        print(f"COOKIE FILE NAME: {comb}")

                        # subprocess to renew the cookie
                        subprocess.run(
                            [
                                "uv",
                                "run",
                                "browser_env/auto_login.py",
                                "--auth_folder",
                                temp_dir,
                                "--site_list",
                                *comb,
                            ]
                        )
                        _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                        print(f"STORAGE STATE: {_c['storage_state']}")
                        assert os.path.exists(_c["storage_state"])
                        # update the config file
                        config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                        with open(config_file, "w") as f:
                            json.dump(_c, f)

                    # Load input images for the task, if any.
                    if image_paths is not None:
                        if isinstance(image_paths, str):
                            image_paths = [image_paths]
                        for image_path in image_paths:
                            # Load image either from the web or from a local path.
                            if image_path.startswith("http"):
                                headers = {
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                                }
                                input_image = Image.open(
                                    requests.get(
                                        image_path, stream=True, headers=headers
                                    ).raw
                                )
                            else:
                                input_image = Image.open(image_path)

                            images.append(input_image)

                logger.info(f"[Config file]: {config_file}")
                logger.info(f"[Intent]: {intent}")

                agent.reset(config_file)
                trajectory = []
                obs, info = env.reset(options={"config_file": config_file})
                state_info: StateInfo = {"observation": obs, "info": info}
                trajectory.append(state_info)

                meta_data = {"action_history": ["None"]}

                while True:
                    early_stop_flag, stop_info = early_stop(
                        trajectory, max_steps, early_stop_thresholds
                    )

                    if early_stop_flag:
                        logger.debug(f"Early stop: {stop_info}")
                        action = create_stop_action(f"Early stop: {stop_info}")
                    else:
                        logger.debug("Generating next action...")
                        try:
                            action, step_usage = agent.next_action(
                                trajectory,
                                intent,
                                images=images,
                                meta_data=meta_data,
                            )
                            task_usage += step_usage
                        except ValueError as e:
                            # get the error message
                            logger.error(f"ValueError in agent.next_action: {e}")
                            action = create_stop_action(f"ERROR: {str(e)}")

                        logger.debug("Action generated.")

                    trajectory.append(action)

                    action_str = get_action_description(
                        action,
                        state_info["info"]["observation_metadata"],
                        action_set_tag=args.action_set_tag,
                        prompt_constructor=agent.prompt_constructor
                        if isinstance(agent, PromptAgent)
                        else None,
                        current_url=state_info["info"]["page"].url,
                    )
                    render_helper.render(
                        action, state_info, meta_data, args.render_screenshot
                    )
                    meta_data["action_history"].append(action_str)

                    if action["action_type"] == ActionTypes.STOP:
                        break

                    obs, _, terminated, _, info = env.step(action)
                    state_info = {"observation": obs, "info": info}
                    trajectory.append(state_info)

                    if terminated:
                        # add a action place holder
                        trajectory.append(create_stop_action(""))
                        break

                # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
                evaluator = evaluator_router(
                    config_file, captioning_fn=eval_caption_image_fn
                )
                score = evaluator(
                    trajectory=trajectory, config_file=config_file, page=env.page
                )

                scores.append(score)

                if score == 1:
                    logger.info(f"[Result] (PASS) {config_file}")
                else:
                    logger.info(f"[Result] (FAIL) {config_file}")

                if args.save_trace_enabled:
                    env.save_trace(Path(args.result_dir) / "traces" / f"{task_id}.zip")

            except openai.OpenAIError as e:
                logger.error(f"[OpenAI Error] {repr(e)}")
            except Exception as e:
                logger.error(f"[Unhandled Error] {repr(e)}]")
                import traceback

                # write to error file
                with open(Path(args.result_dir) / "error.txt", "a") as f:
                    f.write(f"[Config file]: {config_file}\n")
                    f.write(f"[Unhandled Error] {repr(e)}\n")
                    f.write(traceback.format_exc())  # write stack trace to file

            if exp:
                num_steps = (len(trajectory) - 1) / 2 if trajectory else 0
                latency_seconds = time.time() - task_start_time

                # Update cumulative metrics
                cumulative["cumulative_tasks"] += 1
                cumulative["cumulative_successes"] += int(score == 1.0)
                cumulative["cumulative_success_rate"] = (
                    cumulative["cumulative_successes"] / cumulative["cumulative_tasks"]
                )
                cumulative["cumulative_total_tokens"] += task_usage.total_tokens
                cumulative["cumulative_input_tokens"] += task_usage.input_tokens
                cumulative["cumulative_output_tokens"] += task_usage.output_tokens
                cumulative["cumulative_cache_read_tokens"] += task_usage.cache_read_tokens
                cumulative["cumulative_cache_write_tokens"] += task_usage.cache_write_tokens
                cumulative["cumulative_num_steps"] += int(num_steps)
                cumulative["cumulative_latency_seconds"] += latency_seconds

                metrics = TaskMetrics(
                    success=score == 1.0,
                    site=task_site,
                    total_tokens=task_usage.total_tokens,
                    input_tokens=task_usage.input_tokens,
                    output_tokens=task_usage.output_tokens,
                    cache_read_tokens=task_usage.cache_read_tokens,
                    cache_write_tokens=task_usage.cache_write_tokens,
                    num_steps=int(num_steps),
                    latency_seconds=latency_seconds,
                    task_id=task_id,
                    **cumulative,
                )
                exp.log_metrics(metrics)

                logger.info(
                    f"[Cumulative] tasks={cumulative['cumulative_tasks']}, "
                    f"successes={cumulative['cumulative_successes']}, "
                    f"rate={cumulative['cumulative_success_rate']:.3f}"
                )

            render_helper.close()

        if exp:
            exp.finish_wandb_run()

    if args.experiment_name:
        with Experiment(
            name=args.experiment_name,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        ) as exp:
            _run_tasks(exp)
    else:
        _run_tasks(None)

    env.close()
    if len(scores):
        logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [os.path.basename(f).split(".")[0].split("_")[1] for f in result_files]
    logger.debug(f"Task IDs: {task_ids}")
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    test(args, test_file_list)
