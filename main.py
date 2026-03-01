"""
基于光流的图像对齐主入口

支持 FlowFormer 和 FlowFormer++ 算法对两幅图像进行光流估计与对齐，
包含可视化、对比模式、对齐质量评估等功能。
"""

import argparse
import sys
import time

from loguru import logger

from config import config_from_args as get_config
from src.io_utils import load_image, build_output_dir, save_alignment_results, save_flow_data, save_comparison_report, save_evaluation_results, save_image
from src.flow_estimator import FlowEstimator
from src.image_aligner import ImageAligner
from src.visualizer import FlowVisualizer
from src.evaluator import AlignmentEvaluator
from src.comparator import AlgorithmComparator
from download_models import check_models


from config import build_parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    config = get_config(args)

    if not config["image1_path"] or not config["image2_path"]:
        logger.error("必须指定输入图像: --image1 和 --image2")
        sys.exit(1)

    if not check_models(config):
        sys.exit(1)

    img1 = load_image(config["image1_path"])
    img2 = load_image(config["image2_path"])
    
    if img1 is None or img2 is None:
        logger.error("图像读取失败，退出程序.")
        sys.exit(1)

    output_sub_dir = build_output_dir(config["output_dir"], config["image1_path"], config["image2_path"])
    logger.info(f"输出目录: {output_sub_dir}")

    aligner = ImageAligner()
    evaluator = AlignmentEvaluator(config.get("evaluation_metrics")) if config.get("enable_evaluation", False) else None
    visualizer = FlowVisualizer(config.get("visualization_methods")) if config.get("enable_visualization", False) else None

    if config.get("comparison_mode", False) and len(config["algorithms"]) > 1:
        logger.info(f"启动对比模式: {config['algorithms']}")
        comparator = AlgorithmComparator(config)
        report, flows, aligned_images = comparator.run_comparison(
            algorithms=config["algorithms"],
            flow_estimator_factory=FlowEstimator,
            img1=img1,
            img2=img2,
            image_aligner=aligner,
            evaluator=evaluator
        )
        
        if config.get("comparison_mode", False):
            save_comparison_report(output_sub_dir, report, config.get("output_format", "png"))
            
        # 在父目录下保存这四张图像，方便用户直接查看
        ext = config.get("output_format", "png")
        save_image(img1, str(output_sub_dir / f"0_fixed_img1.{ext}"))
        save_image(img2, str(output_sub_dir / f"2_moving_img2.{ext}"))
        for algo in config["algorithms"]:
            if algo in aligned_images:
                save_image(aligned_images[algo], str(output_sub_dir / f"1_aligned_img2_{algo}.{ext}"))
            
        for algo in config["algorithms"]:
            algo_dir = output_sub_dir / algo
            algo_dir.mkdir(exist_ok=True)
            
            save_alignment_results(
                algo_dir,
                img1=img1,
                aligned_img2=aligned_images[algo],
                img2=img2,
                output_format=config.get("output_format", "png")
            )
            
            vis_flows = visualizer.visualize(flows[algo], image=img1) if visualizer else {}
            if config.get("save_flow_data", False):
                flow_vis_img = list(vis_flows.values())[0] if vis_flows else None
                save_flow_data(algo_dir, flows[algo], flow_vis_img, output_format=config.get("output_format", "png"))
                
            if visualizer:
                for method, vis_img in vis_flows.items():
                    save_image(vis_img, str(algo_dir / f"flow/flow_vis_{method}.{config.get('output_format', 'png')}"))
                    
            if evaluator:
                algo_eval = report["results"][algo].get("evaluation")
                if algo_eval:
                    vis_eval = evaluator.generate_visual_evaluation(img1, aligned_images[algo])
                    save_evaluation_results(algo_dir, algo_eval["metrics"], vis_eval, config.get("output_format", "png"))
                
    else:
        algo = config["algorithms"][0]
        logger.info(f"运行算法: {algo}")
        
        estimator = FlowEstimator(algo, config)
        estimator.load_model()
        
        start_time = time.perf_counter()
        flow = estimator.estimate(img1, img2)
        end_time = time.perf_counter()
        logger.info(f"光流估计耗时: {end_time - start_time:.4f}s")
        
        aligned_img2 = aligner.align(img2, flow)
        
        save_alignment_results(
            output_sub_dir,
            img1=img1,
            aligned_img2=aligned_img2,
            img2=img2,
            output_format=config.get("output_format", "png")
        )
        
        vis_flows = visualizer.visualize(flow, image=img1) if visualizer else {}
        if config.get("save_flow_data", False):
            flow_vis_img = list(vis_flows.values())[0] if vis_flows else None
            save_flow_data(output_sub_dir, flow, flow_vis_img, output_format=config.get("output_format", "png"))
            
        if visualizer:
            for method, vis_img in vis_flows.items():
                save_image(vis_img, str(output_sub_dir / f"flow/flow_vis_{method}.{config.get('output_format', 'png')}"))
                
        if evaluator:
            result = evaluator.evaluate(img1, aligned_img2, img2)
            vis_eval = evaluator.generate_visual_evaluation(img1, aligned_img2)
            save_evaluation_results(output_sub_dir, result["metrics"], vis_eval, config.get("output_format", "png"))


if __name__ == "__main__":
    main()
