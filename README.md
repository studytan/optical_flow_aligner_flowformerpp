基于FlowFormer++光流算法的图像对齐工具，支持FlowFormer和FlowFormer++两种算法

demo:
C:\ProgramData\miniconda3\envs\flow\python.exe D:\MyCode\MyCodeProject\PhthonProject\imageAlign\img_align_by_flow\main.py --image1 test_images/0_fixed_image.png --image2 test_images/2_moving_image.png --algorithm flowformer flowformerpp --comparison --evaluate --visualize --save-flow 
2026-03-01 22:24:38.688 | INFO     | src.io_utils:load_image:44 - 已加载图像: 0_fixed_image.png (1024x436)
2026-03-01 22:24:38.699 | INFO     | src.io_utils:load_image:44 - 已加载图像: 2_moving_image.png (1024x436)
2026-03-01 22:24:38.699 | INFO     | __main__:main:46 - 输出目录: output\0_fixed_image_vs_2_moving_image
2026-03-01 22:24:38.700 | INFO     | __main__:main:53 - 启动对比模式: ['flowformer', 'flowformerpp']
2026-03-01 22:24:38.700 | INFO     | src.comparator:run_comparison:54 - 
==================================================
2026-03-01 22:24:38.700 | INFO     | src.comparator:run_comparison:55 - 运行算法: flowformer
2026-03-01 22:24:38.700 | INFO     | src.comparator:run_comparison:56 - ==================================================
2026-03-01 22:24:38.700 | INFO     | src.flow_estimator:load_model:69 - [flowformer] 正在加载模型: checkpoints/flowformer/things.pth
C:\ProgramData\miniconda3\envs\flow\lib\site-packages\timm\models\layers\__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
C:\ProgramData\miniconda3\envs\flow\lib\site-packages\timm\models\registry.py:4: FutureWarning: Importing from timm.models.registry is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
C:\ProgramData\miniconda3\envs\flow\lib\site-packages\timm\models\helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
2026-03-01 22:24:43.164 | INFO     | src.flow_estimator:load_model:133 - [flowformer] 模型加载成功 (Device: cuda)
2026-03-01 22:24:43.169 | INFO     | src.flow_estimator:estimate:155 - [flowformer] 推理中 (Tile=True, FP16=True)...
2026-03-01 22:24:50.155 | INFO     | src.comparator:run_comparison:76 - 推理时间: 6.9903s
2026-03-01 22:24:50.155 | INFO     | src.comparator:run_comparison:82 - GPU 显存峰值: 1909.68 MB
2026-03-01 22:24:50.158 | INFO     | src.image_aligner:align:60 - 图像对齐完成: 输出尺寸 1024x436
2026-03-01 22:24:50.159 | INFO     | src.evaluator:evaluate:193 - 计算对齐质量指标...
2026-03-01 22:24:50.291 | INFO     | src.evaluator:evaluate:198 -   SSIM: 0.939122
2026-03-01 22:24:50.647 | INFO     | src.evaluator:evaluate:198 -   PSNR: 27.866278
2026-03-01 22:24:50.658 | INFO     | src.evaluator:evaluate:198 -   MSE: 106.280241
2026-03-01 22:24:50.669 | INFO     | src.evaluator:evaluate:198 -   MAE: 3.202957
2026-03-01 22:24:50.677 | INFO     | src.evaluator:evaluate:198 -   NCC: 0.989434
2026-03-01 22:24:50.677 | INFO     | src.evaluator:evaluate:207 - 计算对齐前指标 (用于对比)...
2026-03-01 22:24:50.735 | INFO     | src.evaluator:evaluate:216 -   SSIM: 0.706713 -> 0.939122 (↑ 0.232409)
2026-03-01 22:24:50.748 | INFO     | src.evaluator:evaluate:216 -   PSNR: 22.885441 -> 27.866278 (↑ 4.980837)
2026-03-01 22:24:50.758 | INFO     | src.evaluator:evaluate:216 -   MSE: 334.607919 -> 106.280241 (↓ 228.327678)
2026-03-01 22:24:50.770 | INFO     | src.evaluator:evaluate:216 -   MAE: 9.164684 -> 3.202957 (↓ 5.961727)
2026-03-01 22:24:50.776 | INFO     | src.evaluator:evaluate:216 -   NCC: 0.967058 -> 0.989434 (↑ 0.022376)
2026-03-01 22:24:50.795 | INFO     | src.comparator:run_comparison:54 - 
==================================================
2026-03-01 22:24:50.795 | INFO     | src.comparator:run_comparison:55 - 运行算法: flowformerpp
2026-03-01 22:24:50.795 | INFO     | src.comparator:run_comparison:56 - ==================================================
2026-03-01 22:24:50.796 | INFO     | src.flow_estimator:load_model:69 - [flowformerpp] 正在加载模型: checkpoints/flowformerpp/things.pth
[Using larger cost as gt, radius is 15]
[Decoder flow_or_pe setting is: and]
[Using GMA]
2026-03-01 22:24:53.283 | INFO     | src.flow_estimator:load_model:133 - [flowformerpp] 模型加载成功 (Device: cuda)
2026-03-01 22:24:53.287 | INFO     | src.flow_estimator:estimate:155 - [flowformerpp] 推理中 (Tile=True, FP16=True)...
2026-03-01 22:24:57.945 | INFO     | src.comparator:run_comparison:76 - 推理时间: 4.6608s
2026-03-01 22:24:57.945 | INFO     | src.comparator:run_comparison:82 - GPU 显存峰值: 1911.94 MB
2026-03-01 22:24:57.948 | INFO     | src.image_aligner:align:60 - 图像对齐完成: 输出尺寸 1024x436
2026-03-01 22:24:57.949 | INFO     | src.evaluator:evaluate:193 - 计算对齐质量指标...
2026-03-01 22:24:58.007 | INFO     | src.evaluator:evaluate:198 -   SSIM: 0.941137
2026-03-01 22:24:58.018 | INFO     | src.evaluator:evaluate:198 -   PSNR: 28.201394
2026-03-01 22:24:58.028 | INFO     | src.evaluator:evaluate:198 -   MSE: 98.387728
2026-03-01 22:24:58.039 | INFO     | src.evaluator:evaluate:198 -   MAE: 3.122397
2026-03-01 22:24:58.046 | INFO     | src.evaluator:evaluate:198 -   NCC: 0.990232
2026-03-01 22:24:58.046 | INFO     | src.evaluator:evaluate:207 - 计算对齐前指标 (用于对比)...
2026-03-01 22:24:58.102 | INFO     | src.evaluator:evaluate:216 -   SSIM: 0.706713 -> 0.941137 (↑ 0.234424)
2026-03-01 22:24:58.113 | INFO     | src.evaluator:evaluate:216 -   PSNR: 22.885441 -> 28.201394 (↑ 5.315953)
2026-03-01 22:24:58.124 | INFO     | src.evaluator:evaluate:216 -   MSE: 334.607919 -> 98.387728 (↓ 236.220191)
2026-03-01 22:24:58.134 | INFO     | src.evaluator:evaluate:216 -   MAE: 9.164684 -> 3.122397 (↓ 6.042286)
2026-03-01 22:24:58.141 | INFO     | src.evaluator:evaluate:216 -   NCC: 0.967058 -> 0.990232 (↑ 0.023174)
2026-03-01 22:24:58.167 | INFO     | src.comparator:run_comparison:117 - 光流差异 (flowformer_vs_flowformerpp): EPE=0.0871
2026-03-01 22:24:58.167 | INFO     | src.comparator:_print_comparison_summary:126 - 
============================================================
2026-03-01 22:24:58.167 | INFO     | src.comparator:_print_comparison_summary:127 - 对比汇总
2026-03-01 22:24:58.167 | INFO     | src.comparator:_print_comparison_summary:128 - ============================================================
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:131 - 
[flowformer]
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:133 -   推理时间: 6.9903s
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:135 -   GPU 显存峰值: 1909.68 MB
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   SSIM: 0.939122
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   PSNR: 27.866278
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   MSE: 106.280241
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   MAE: 3.202957
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   NCC: 0.989434
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:131 - 
[flowformerpp]
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:133 -   推理时间: 4.6608s
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:135 -   GPU 显存峰值: 1911.94 MB
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   SSIM: 0.941137
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   PSNR: 28.201394
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   MSE: 98.387728
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   MAE: 3.122397
2026-03-01 22:24:58.168 | INFO     | src.comparator:_print_comparison_summary:138 -   NCC: 0.990232
2026-03-01 22:24:58.172 | INFO     | src.io_utils:save_comparison_report:185 - 对比报告已保存: output\0_fixed_image_vs_2_moving_image\comparison\comparison_report.json
2026-03-01 22:24:58.196 | INFO     | src.io_utils:save_image:62 - 已保存图像: 0_fixed_img1.png
2026-03-01 22:24:58.227 | INFO     | src.io_utils:save_image:62 - 已保存图像: 2_moving_img2.png
2026-03-01 22:24:58.257 | INFO     | src.io_utils:save_image:62 - 已保存图像: 1_aligned_img2_flowformer.png
2026-03-01 22:24:58.280 | INFO     | src.io_utils:save_image:62 - 已保存图像: 1_aligned_img2_flowformerpp.png
2026-03-01 22:24:58.303 | INFO     | src.io_utils:save_image:62 - 已保存图像: 0_fixed_img1.png
2026-03-01 22:24:58.327 | INFO     | src.io_utils:save_image:62 - 已保存图像: 1_aligned_img2.png
2026-03-01 22:24:58.354 | INFO     | src.io_utils:save_image:62 - 已保存图像: 2_moving_img2.png
2026-03-01 22:24:58.355 | INFO     | src.io_utils:save_alignment_results:118 - 对齐结果已保存到: output\0_fixed_image_vs_2_moving_image\flowformer
2026-03-01 22:24:58.355 | INFO     | src.visualizer:visualize:235 - 光流可视化: color_wheel
2026-03-01 22:24:58.464 | INFO     | src.io_utils:save_flow_data:137 - 光流 map 已保存: flow_map.npy (shape=(436, 1024, 2))
2026-03-01 22:24:58.475 | INFO     | src.io_utils:save_image:62 - 已保存图像: flow_vis.png
2026-03-01 22:24:58.487 | INFO     | src.io_utils:save_image:62 - 已保存图像: flow_vis_color_wheel.png
2026-03-01 22:24:58.487 | INFO     | src.evaluator:generate_visual_evaluation:236 - 生成差异热力图...
2026-03-01 22:24:58.498 | INFO     | src.evaluator:generate_visual_evaluation:239 - 生成棋盘格叠加图...
2026-03-01 22:24:58.501 | INFO     | src.io_utils:save_evaluation_results:164 - 评估指标已保存: output\0_fixed_image_vs_2_moving_image\flowformer\evaluation\metrics.json
2026-03-01 22:24:58.519 | INFO     | src.io_utils:save_image:62 - 已保存图像: diff_heatmap.png
2026-03-01 22:24:58.542 | INFO     | src.io_utils:save_image:62 - 已保存图像: checkerboard.png
2026-03-01 22:24:58.566 | INFO     | src.io_utils:save_image:62 - 已保存图像: 0_fixed_img1.png
2026-03-01 22:24:58.589 | INFO     | src.io_utils:save_image:62 - 已保存图像: 1_aligned_img2.png
2026-03-01 22:24:58.613 | INFO     | src.io_utils:save_image:62 - 已保存图像: 2_moving_img2.png
2026-03-01 22:24:58.613 | INFO     | src.io_utils:save_alignment_results:118 - 对齐结果已保存到: output\0_fixed_image_vs_2_moving_image\flowformerpp
2026-03-01 22:24:58.614 | INFO     | src.visualizer:visualize:235 - 光流可视化: color_wheel
2026-03-01 22:24:58.702 | INFO     | src.io_utils:save_flow_data:137 - 光流 map 已保存: flow_map.npy (shape=(436, 1024, 2))
2026-03-01 22:24:58.716 | INFO     | src.io_utils:save_image:62 - 已保存图像: flow_vis.png
2026-03-01 22:24:58.729 | INFO     | src.io_utils:save_image:62 - 已保存图像: flow_vis_color_wheel.png
2026-03-01 22:24:58.729 | INFO     | src.evaluator:generate_visual_evaluation:236 - 生成差异热力图...
2026-03-01 22:24:58.739 | INFO     | src.evaluator:generate_visual_evaluation:239 - 生成棋盘格叠加图...
2026-03-01 22:24:58.743 | INFO     | src.io_utils:save_evaluation_results:164 - 评估指标已保存: output\0_fixed_image_vs_2_moving_image\flowformerpp\evaluation\metrics.json
2026-03-01 22:24:58.768 | INFO     | src.io_utils:save_image:62 - 已保存图像: diff_heatmap.png
2026-03-01 22:24:58.795 | INFO     | src.io_utils:save_image:62 - 已保存图像: checkerboard.png

进程已结束，退出代码为 0
