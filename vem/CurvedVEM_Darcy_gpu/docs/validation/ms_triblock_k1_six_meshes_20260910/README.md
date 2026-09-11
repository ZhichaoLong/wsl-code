# MS 4 六层实验验证证据

状态：按用户要求停止。前五层已完成，128×128 最后完整记录为 25/128 步；六层完整验收未执行。

主程序 `make -j4 MS_triblock_bdf2` 编译，运行 `./build/MS_triblock_bdf2`。k=1，T=0.1，实际 4×4 至 128×128；dt_target=0.1h。求解设置及每层结果见 `../../../output/MS_triblock_bdf2_k1/`。初次 restart=80 在 32×32 失败，完整旧记录在 `../../../output/MS_triblock_bdf2_k1_restart80_failed/`；正式重跑使用 restart=320、maxit=20000，容差不变。

`source_binary_sha256.json` 记录本次源码和二进制，最终测试日志将在全程完成后保存。
