import os
import time
import argparse
from pathlib import Path
import sys
import subprocess
from typing import Optional

print(sys.executable)
print("SYS.PATH[0:3] =", sys.path[:3])

ROOT = Path(__file__).resolve().parent  # GE/
sys.path.insert(0, str(ROOT / "InstantMesh"))
sys.path.insert(0, str(ROOT / "LOD"))

import run
import lod


def open_file(path: str):
    try:
        if path and os.path.exists(path):
            os.startfile(path)  # Windows 전용
    except Exception:
        pass


def prompt_choice(msg: str, valid: set[str]) -> str:
    while True:
        v = input(msg).strip().upper()
        if v in valid:
            return v
        print(f"잘못된 입력. 가능한 값: {', '.join(sorted(valid))}")


def prompt_path(msg: str, must_exist: bool = True) -> str:
    while True:
        raw = input(msg).strip()
        if not raw:
            print("값이 비어 있음. 다시 입력.")
            continue

        p = raw.strip().strip('"').strip("'")
        p = str(Path(p).expanduser())

        if must_exist and not os.path.exists(p):
            print(f"경로가 존재하지 않음: {p}")
            continue
        return p


def resolve_default_config(model: str) -> Optional[str]:
    """
    InstantMesh/configs 내부에서 base/large에 해당하는 yaml 자동 탐색.
    """
    cfg_dir = ROOT / "InstantMesh" / "configs"
    candidates = [
        cfg_dir / f"{model}.yaml",
        cfg_dir / f"{model}.yml",
        cfg_dir / f"instantmesh_{model}.yaml",
        cfg_dir / f"instantmesh_{model}.yml",
        cfg_dir / f"{model}_config.yaml",
        cfg_dir / f"{model}_config.yml",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    if cfg_dir.exists():
        yamls = list(cfg_dir.glob("*.yml")) + list(cfg_dir.glob("*.yaml"))
        for y in yamls:
            if model.lower() in y.name.lower():
                return str(y.resolve())

    return None


def ask_processing_choice(input_file: str) -> str:
 
    while True:
        print(f"\n[질문 1] 입력된 모델({input_file})의 상태는 무엇입니까?")
        print("  1. Low Poly (거칠고 각짐)")
        print("  2. High Poly (무겁고 복잡함)")
        choice_type = input("  >>> 선택 (1 또는 2): ").strip()

        if choice_type == "1":
            print("\n[질문 2] Low Poly 모델을 어떻게 처리할까요?")
            print("  A. 단순히 정리만 하기 (Raw Mesh Clean)")
            print("  B. 고해상도로 업그레이드 (Refinement to High Poly)")
            choice_action = input("  >>> 선택 (A 또는 B): ").strip().upper()

            if choice_action == "A":
                return "low_clean"
            if choice_action == "B":
                return "low_refine"

            print("잘못된 입력입니다.")
            continue

        if choice_type == "2":
            print("\n[질문 2] 어떤 LOD 단계로 줄일까요?")
            print("  L. Low Poly (모바일용, 2,000 tris)")
            print("  M. Mid Poly (PC용, 10,000 tris)")
            print("  H. High Poly (시네마틱용, 50,000 tris)")
            print("  A. 전부 다 생성 (All)")
            choice_lod = input("  >>> 선택 (L, M, H, A): ").strip().upper()

            if choice_lod == "L":
                return "high_low"
            if choice_lod == "M":
                return "high_mid"
            if choice_lod == "H":
                return "high_high"
            if choice_lod == "A":
                return "high_all"

            print("잘못된 입력입니다.")
            continue

        print("잘못된 입력입니다. 1 또는 2를 입력하세요.")


def build_pipeline_parser():
    p = argparse.ArgumentParser("InstantMesh → (preview video) → LOD pipeline")

    p.add_argument("--image", help="입력 이미지 경로 (또는 폴더). 없으면 프롬프트")
    p.add_argument("--model", choices=["base", "large"], help="base/large. 없으면 프롬프트")
    p.add_argument("--config", help="config yaml 직접 지정(선택). 없으면 model로 자동 탐색")
    p.add_argument("--out", default="pipeline_output", help="전체 출력 베이스 폴더")

    # InstantMesh 옵션
    p.add_argument("--diffusion_steps", type=int, default=75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--distance", type=float, default=4.5)
    p.add_argument("--view", type=int, default=6, choices=[4, 6])
    p.add_argument("--no_rembg", action="store_true")

    #  기본값: texmap ON
    p.add_argument("--export_texmap", dest="export_texmap", action="store_true", default=True)
    p.add_argument("--no_export_texmap", dest="export_texmap", action="store_false")

    # 기본값: mp4 생성 + 자동 열기 ON
    p.add_argument("--preview_video", dest="preview_video", action="store_true", default=True)
    p.add_argument("--no_preview_video", dest="preview_video", action="store_false")
    p.add_argument("--auto_open_video", dest="auto_open_video", action="store_true", default=True)
    p.add_argument("--no_auto_open_video", dest="auto_open_video", action="store_false")

    #  기본값: 영상 보고 질문(=interactive) ON
    p.add_argument("--interactive", dest="interactive", action="store_true", default=True)
    p.add_argument("--no_interactive", dest="interactive", action="store_false")

    # non-interactive용 (원할 때만 사용)
    p.add_argument("--mode", choices=["low", "high"], help="non-interactive 강제: low/high")
    p.add_argument("--low_action", choices=["clean", "refine"], default="clean")
    p.add_argument("--lod", choices=["low", "mid", "high", "all"], default="all")

    p.add_argument("--export_fmt", choices=["glb", "obj"], default="glb", help="LOD 결과 포맷(권장 glb)")

    return p


def print_repro_cmd(args, config_path: str, input_path: str):
    parts = [
        "python", str(Path(__file__).name),
        "--image", f"\"{input_path}\"",
        "--out", f"\"{args.out}\"",
        "--diffusion_steps", str(args.diffusion_steps),
        "--seed", str(args.seed),
        "--scale", str(args.scale),
        "--distance", str(args.distance),
        "--view", str(args.view),
        "--export_fmt", str(args.export_fmt),
    ]

    # model/config 표기
    if args.config:
        parts += ["--config", f"\"{config_path}\""]
    elif args.model:
        parts += ["--model", args.model]
    else:
        parts += ["--config", f"\"{config_path}\""]

    if args.no_rembg:
        parts.append("--no_rembg")

    if args.export_texmap:
        parts.append("--export_texmap")
    else:
        parts.append("--no_export_texmap")

    if args.preview_video:
        parts.append("--preview_video")
    else:
        parts.append("--no_preview_video")

    if args.auto_open_video:
        parts.append("--auto_open_video")
    else:
        parts.append("--no_auto_open_video")

    if args.interactive:
        parts.append("--interactive")
    else:
        parts.append("--no_interactive")

    # non-interactive 강제 모드면 추가
    if args.mode:
        parts += ["--mode", args.mode]
        if args.mode == "low":
            parts += ["--low_action", args.low_action]
        if args.mode == "high":
            parts += ["--lod", args.lod]

    print("\n[Repro Command]")
    print(" ".join(parts))
    print()


def main():
    args = build_pipeline_parser().parse_args()

    # 1) 이미지 경로 입력받기
    if not args.image:
        args.image = prompt_path("입력 이미지(또는 폴더) 경로를 입력: ", must_exist=True)
    input_path = str(Path(args.image).resolve())
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"image/dir 없음: {input_path}")

    # 2) config 결정
    if args.config:
        config_path = str(Path(args.config).resolve())
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config 없음: {config_path}")
    else:
        if not args.model:
            m = prompt_choice("InstantMesh 모델 선택: BASE(B) / LARGE(L): ", {"B", "L"})
            args.model = "base" if m == "B" else "large"

        auto_cfg = resolve_default_config(args.model)
        if not auto_cfg:
            print(f"[Config Auto] model={args.model}에 해당하는 yaml을 {ROOT / 'InstantMesh' / 'configs'}에서 못 찾음")
            auto_cfg = prompt_path("config yaml 경로를 직접 입력: ", must_exist=True)

        config_path = str(Path(auto_cfg).resolve())
        print(f"[Config] {config_path}")

    # 3) 재현 커맨드 출력 (선택값 포함)
    print_repro_cmd(args, config_path, input_path)

    # 4) job 폴더
    job_id = time.strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.out).resolve()
    job_dir = base_out / f"job_{job_id}"
    instant_out = job_dir / "instantmesh"
    final_out = job_dir / "final_meshes"
    instant_out.mkdir(parents=True, exist_ok=True)
    final_out.mkdir(parents=True, exist_ok=True)

    # 5) InstantMesh 실행
    ro_parser = run.build_parser()
    save_video_flag = bool(args.preview_video) or bool(args.auto_open_video)

    ro_args_list = [
        config_path,
        input_path,
        "--output_path", str(instant_out),
        "--diffusion_steps", str(args.diffusion_steps),
        "--seed", str(args.seed),
        "--scale", str(args.scale),
        "--distance", str(args.distance),
        "--view", str(args.view),
    ]
    if args.no_rembg:
        ro_args_list.append("--no_rembg")
    if args.export_texmap:
        ro_args_list.append("--export_texmap")
    if save_video_flag:
        ro_args_list.append("--save_video")

    ro_args = ro_parser.parse_args(ro_args_list)

    print("\n[1/3] InstantMesh 실행 시작...")
    result = run.run(ro_args)

    mesh_paths = result.get("mesh_paths", [])
    video_paths = result.get("video_paths", [])

    if not mesh_paths:
        raise RuntimeError("메쉬 생성 결과(mesh_paths)가 비어 있음")

    print("\n생성 완료!")
    print(f"- job_dir: {job_dir}")
    print(f"- meshes: {mesh_paths}")
    if video_paths:
        print(f"- video: {video_paths[0]}")

    # 6) mp4 자동 열기
    if video_paths and args.auto_open_video:
        open_file(video_paths[0])

    # 7) 사용자 선택(영상 보고 결정)
    print("\n[2/3] 사용자 입력")

    if args.mode:
        # non-interactive 강제
        if args.mode == "low":
            chosen = "low_clean" if args.low_action == "clean" else "low_refine"
        else:
            chosen = f"high_{args.lod}"  # high_low/high_mid/high_high/high_all
    else:
        # interactive (기본)
        input_file_for_prompt = Path(mesh_paths[0]).name
        # lod_original에 같은 함수가 있으면 그걸 쓰고, 없으면 로컬 함수 사용
        if hasattr(lod, "ask_processing_choice"):
            chosen = lod.ask_processing_choice(input_file_for_prompt)
        else:
            chosen = ask_processing_choice(input_file_for_prompt)

    # 8) LOD 실행
    print("\n[3/3] LOD 실행")
    for obj_path in mesh_paths:
        obj_name = Path(obj_path).stem
        per_mesh_out = final_out / obj_name
        per_mesh_out.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {obj_path}")
        print(f"Output dir: {per_mesh_out}")

        if chosen == "low_clean":
            lod.run_simple_clean(obj_path, str(per_mesh_out))
        elif chosen == "low_refine":
            lod.run_academic_upsampling(obj_path, str(per_mesh_out))
        else:
            mode_map = {
                "high_low": "LOW",
                "high_mid": "MID",
                "high_high": "HIGH",
                "high_all": "ALL",
            }
            lod.run_smart_decimation(obj_path, str(per_mesh_out), mode_map[chosen])

    print("\n==============================================")
    print("✓ 전체 파이프라인 완료")
    print(f"최종 결과 폴더: {final_out}")
    print("==============================================")


if __name__ == "__main__":
    main()
