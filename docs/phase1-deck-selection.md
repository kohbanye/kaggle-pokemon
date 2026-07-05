# Phase 1 — デッキ選定（評価基盤つき）

PLAN.md の **Phase 1（最大のレバー＝デッキ選択）** を回すための評価基盤と手順。
「エージェント固定・デッキ変更」で総当たりし、**Wilson 95%信頼区間**で有意差の
出たデッキだけを残す（PLAN §A/§B の規律）。

## いま出来ていること（データ不要・テスト済み）

| モジュール | 役割 |
|---|---|
| `src/eval/stats.py` | Wilson信頼区間・`WinRate`集計・対50%の有意判定 |
| `src/eval/schedule.py` | 総当たり＋**先後入替の paired seed** スケジュール（分散低減） |
| `src/eval/deck.py` | デッキ読込＋合法性（60枚／同名4枚・基本エネ除外／ACE SPEC1枚） |
| `src/eval/runner.py` | エンジン非依存の対戦ランナー（`Engine`/`Agent` protocol, `play_match`, `aggregate`） |
| `agents/random_agent.py` | ランダム合法手ベースライン（`sim_smoke.py`検証済みロジック） |
| `scripts/run_tournament.py` | 上記を束ねる Phase 1 CLI（cg アダプタ＝スキーマの唯一の依存点） |

`tests/test_{stats,schedule,deck,runner}.py` が純ロジックを網羅（`FakeEngine`で
ランナーの routing/終了/集計を検証）。`uv run pytest -q` で 31 tests green。

## まだ出来ないこと（ブロッカー）

Phase 1 の**成果物＝実測でのデッキ確定**は、次が揃うまで出せない:

1. **カードデータ＋シミュレータ** … `data/` は gitignore。`bash scripts/download_data.sh`
   に **Kaggle 認証（`~/.kaggle/kaggle.json`）とルール同意**が必要。当環境には未配置。
2. **候補デッキ** … 実アーキタイプのデッキは**カードプール（カードID）が必要**。
   EDA でメタ/相性を見てから `decks/` に定義する（`decks/README.md` 参照）。
3. **cg アダプタのスキーマ確認** … `scripts/run_tournament.py` の `CgEngine._step` は
   「どちらの手番か」を obs から取る前提（`turnPlayer`/`player` を仮置き）。
   `sim_smoke.py` は単一エージェント駆動なので**このキーは未検証**。データ取得後に
   実 obs を見て、必要ならこのアダプタだけ直す（他は obs スキーマ非依存）。

## 実行手順（データが揃ったら）

```bash
uv sync
bash scripts/download_data.sh          # Kaggle認証＋ルール同意が前提
# decks/ に候補デッキ(60枚IDのtxt/csv)を数〜十数個置く
uv run python scripts/run_tournament.py --games-per-pair 40 --seed 0
```

出力は各ペアの `勝率 [CI下,CI上]* (W-L-T)`（`*`＝CIが50%を跨がない＝有意）。
x86-64 なので原則 Docker 無しで動く見込み（動かない場合は README の Docker 手順）。

## Exit 基準（PLAN 準拠・再掲）

- 主力デッキ1個＋対抗1〜2個を、相性・安定性の理由つきで確定。
- 「デッキ差」が「ここまでのエージェント差」より勝率を動かすことを定量確認。
- 初回ラダー提出でローカル↔ラダーの対応を取り始める。
- 結果は PLAN.md「採否台帳」に追記。
