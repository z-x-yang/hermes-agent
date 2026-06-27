#!/usr/bin/env bash
# new-worktree.sh <task> [topic] — 开一个开发 worktree。
#
# worktree 只隔离源码。各 worktree 间相同、可复用的资产(venv / 数据集 / 模型等)按
# "可写性"区别对待 —— 判据是"跨 worktree 是否同一份",不是"大不大":
#   - 可写的(venv / node_modules / 构建产物):*不* symlink 到主 checkout —— 否则
#     worktree 里一句 pip/npm install 或一次 build 会顺着 symlink 写穿、污染生产
#     gateway 的依赖(codex review 红线)。Python 测试靠 scripts/run_tests.sh 的
#     venv fallback(只读用主 checkout 的 venv);要装新依赖,就在本 worktree 自建
#     venv —— 那是"改依赖"的特例,本就该隔离。
#   - 只读的(数据集 / 模型权重):列进 RO_SHARED,symlink 共享 —— 省空间,且只读、
#     不会被写穿。
set -euo pipefail

MAIN="$HOME/.hermes/hermes-agent"
# 只读、各 worktree 相同的资产:symlink 共享安全。按 repo 增减;hermes-agent 暂无,留空示例。
RO_SHARED=()   # 例:RO_SHARED=(data models checkpoints)

task="${1:?usage: new-worktree.sh <task> [branch-topic]}"
topic="${2:-$task}"
wt="$HOME/.hermes/worktrees/$task"

cd "$MAIN"
git worktree add "$wt" -b "feat/$topic" main

# 防护展开:bash 3.2 + set -u 下,空数组的 "${arr[@]}" 会被当作 unbound variable 报错。
# "${arr[@]+...}" 形式在数组为空时整体不展开(不触发 unbound),非空时正常带引号展开。
for asset in "${RO_SHARED[@]+"${RO_SHARED[@]}"}"; do
  if [ -e "$MAIN/$asset" ] && [ ! -e "$wt/$asset" ]; then
    ln -s "$MAIN/$asset" "$wt/$asset"
    echo "  shared (read-only): $asset"
  fi
done

echo "✓ worktree: $wt  (branch feat/$topic)"
echo "  Python 测试: scripts/run_tests.sh 会 fallback 用主 venv(只读),无需自建"
echo "  开发完、测试过 → 回主 checkout: git merge --squash feat/$topic && git commit"
echo "  然后回收: git worktree remove $wt && git branch -D feat/$topic"
