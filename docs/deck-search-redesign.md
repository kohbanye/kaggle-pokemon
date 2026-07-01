# デッキ探索リデザイン — 背景と段階プラン

> 本書は **デッキ探索（QD/co-evo）の作り直し**を、ablation 駆動でステップごとに進めるための
> 自己完結ドキュメント。新しいセッションはこの1枚を読めば文脈なしで着手できる。
> 親方針は [PLAN.md](../PLAN.md)（ablation規律・二層評価）に従う。
> 関連メモリ: `qd-niche-redesign`, `ladder-drop-deck-coadapt`, `rl-eval-methodology`,
> `deckbuilding-methods-research`。作業ブランチ: `feat/qd-niche-axes`（ステップごとに派生可）。

---

## 0. 背景（このリデザインに至った診断）

### 0.1 症状：学習するほどラダーが下がる
ラダー実測（提出単位）：

| 提出 | 学習量 | デッキ | ラダーElo |
|---|---|---|---|
| greedy + metal | なし（ヒューリスティック） | metal_aggro | **532.8** ← 最高 |
| recurrent paper net + greedy | BC/OSFP（co-evo前） | greedy系 | 494.1 |
| ismcts（探索） | — | metal_aggro | 440.3 |
| recurrent run3_r6 + grass | co-evo 3run | grass | 451.2 |
| recurrent run7_r6 + run7_best | co-evo 7run | run7_best | 提出済(PENDING) |

「学習が進むほど低い」。原因を本セッションで多角的に診断（`notebooks/02_play_diagnostics.ipynb`）。

### 0.2 真因は2つ（プレイ退化ではない）
- **デッキ側（co-adapt）**：co-evo のデッキは「自分のネット専用にしか強くない」。中立な greedy に
  持たせると弱い：素のgreedyで `run7_best 0.475` / `grass 0.458` vs `metal 0.658`
  （`scripts/deck_strength.py`）。ネットが弱デッキを“化けさせる”が、ラダーの未知相手には移らない。
- **プレイ側（mirror過適合）**：自己対戦ミラーが「**常に後攻**」のような固定癖を学習。
  go-first率＝ greedy/init 1.0、**全co-evoネット 0.0**（`scripts/opening_diag.py`）。
  ※プレイ自体は co-evo で greedy 並みに効率化されている（退化ではない）。

### 0.3 NashConv で定量化（搾取され度＝低いほど頑健）
`scripts/nashconv_eval.py`（11戦略×総当たり）：

| 戦略 | NashConv | 最悪の相手 |
|---|---|---|
| net_run7 \| metal | **0.13**（我々で最頑健） | greedy\|fire |
| greedy \| metal（ラダー最高） | 0.27 | greedyFF\|metal |
| **net_run7 \| run7_best（提出中）** | **0.80**（最も搾取される） | greedy\|fire |
| net_run3 \| grass（451提出） | 0.80 | greedyFF\|metal |

同じネットでデッキを `metal→run7_best` にするだけで `0.13→0.80`（約6倍搾取される）。
Nash混合は `greedy|fire` に収束（fireがmetalを弱点で食う＝metalの炎弱点は実在）。

### 0.4 探索コードの欠陥（`scripts/qd_deck_search.py`）
1. **目的が自己参照**：適応度＝「ネットが両サイドを回したときのメタ勝率」→ ネット専用デッキを選ぶ。
2. **操作が弱い**：`mutate`＝「4枚抜いて一様ランダムで埋める」→ 複数枚の協調変更（ドローエンジン等）が
   できず、メタ近傍を山登り。全デッキがトレーナー薄(17-19)・エネ過多(27-31)で、通常Standard
   （トレーナー22-34/エネ8-15）から乖離＝**まともなビルドを一度も探索していない**＝伸びしろ。
3. **基準が弱くて固定**：相手＝骨格そっくりのtype決定。バーが上がらない。
4. **毎ラウンド作り直し**：改善が複利化しない。

### 0.5 文献調査の結論（deep-research, `deckbuilding-methods-research`）
QD/MAP-Elites は**SOTA**だが、我々の**設定**が標準から外れている。標準対策：
- **目的**：García-Sánchez 2016 の**辞書式適応度**（①合法性→②勝率→③相手間勝率分散の最小化）。
  頑健性は“主目的”でなく“タイブレーク”。→ **NashConvを直接最適化しない**（指標過適合を避ける）。
- **entanglement**：固定パイロット依存は文献も明言「解くのが難しい」→ **複数パイロット評価**で緩和。
- **固定弱相手は退化**（Miernik & Kowalski）→ progressive/coevolution。
- **操作**：自由なカードリスト表現は維持し、**変異だけ heuristic / active-gene 化**
  （空間を縛らず操作を賢く）。
- **高コスト評価 → オンライン深層サロゲート（DSA-ME, GECCO2022 = QDデッキ構築SOTA）/ SAIL**。
- **探索⇄活用ダイヤル**：CMA-MAE の α。
- **coevolution（GAME 2025）**＝品質↑だが多様性↓ → 片側アーカイブで多様性を保つハイブリッド。

---

## 1. 評価方法論（指標とゲート）

二層評価（PLAN.md C 準拠）。**暫定指標＝ローカル（速い・相対比較）／最終ゲート＝ラダー（真値）**。
ローカルがラダーを誤予測することは既知なので、**複数のローカル指標を併用し、ラダーとの相関を蓄積**して
「ラダーを最も予測する指標」を育てる。

### 1.1 ローカル評価スイート（各ステップで回す）
1. **NashConv（暫定の主指標）** — `scripts/nashconv_eval.py`。制限付き＝真の搾取可能性の下限。
   **低いほど頑健**。⚠️ 母集団に**強い搾取役**が要る（弱いと過小評価）。母集団は固定し版管理する。
2. **複数パイロットのデッキ地力** — `scripts/deck_strength.py`。greedy/net で回した field 勝率。
   「誰が回しても強いか」（co-adapt検出）。
3. **held-out 相手への勝率（汎化）** — 学習・適応度・NashConv母集団が**一度も使わない**相手プールを
   別に用意し、それへの勝率＋Wilson CI。ラダーの代理。**※新規に作る（下記 1.4）**。
4. **規範プレイ（非自己参照のサニティ）** — `scripts/opening_diag.py`（go-first率/ベンチ展開/エネ付け）。
   ミラー過適合の癖を検出。
5. **多様性/カバレッジ** — アーカイブのセル被覆数とセル間の勝率分散（強い**かつ**多様か）。

> 集計は Wilson CI＋スロット交換を維持。設定横断は IQM＋bootstrap を推奨（rliable）。

### 1.2 ラダー（最終ゲート、各ステップの採否）
- ⚠️ **ラダーは高分散**（2026-06-29 実測の教訓, [[net-pilot-ladder-bad]]）：**同一エージェント・同一デッキでも ~±200 Elo 振れる**うえ、COMPLETE 後も数時間ドリフトする
  （例：net_run7+metal が 360.3→445.8）。**1提出の数値は「真値」でなくノイズの大きいサンプル**。
  → **単発のラダー値で keep/drop しない**。基準の greedy+metal 532.8 自体も単発ノイズサンプルと見る。
- 採否は **まずローカルの統制比較**（Wilson CI＋スロット交換＋同一相手プール）で決め、ラダーは
  「**±200 より大きく・繰り返し開いた差**」のときだけ確証に使う（必要なら再提出して平均）。
- 既知の（ノイズ込み）最良ラダー基準＝ **greedy+metal ~532.8**。当面これを超えることが North Star。

### 1.3 ラダー相関の蓄積（指標を育てる）
- 提出のたびに `(各ローカル指標, ラダーElo)` を1行記録（`results/ladder_corr.csv` を新設）。
- 5〜8提出たまったら、どのローカル指標がラダーと相関するかを確認 → **主指標を更新**。
  ※ラダーが高分散（§1.2）なので、相関は**多数の点が貯まってから**判断（1〜2点で結論しない）。
- 仮説：NashConv（held-out母集団）と「多パイロット最悪ケース勝率」がラダーと相関する。要実測。

### 1.4 held-out 相手プールの新設（前提作業, どのステップより先 or Step1と同時）
- 学習/適応度/既存NashConv母集団に**含まれない**相手を10〜16個用意：
  多様なtypeデッキ×複数パイロット（greedy・heuristic・別checkpoint）、**先攻型**も含める。
- 用途：上記スイートの③、および NashConv の「搾取役」強化。
- 成果物：`decklists/heldout/` ＋ `scripts/heldout_eval.py`（`deck_strength.py` を流用）。

---

## 2. 現状ベースライン（比較の基準値）

| 指標 | 値（現状） | 出典 |
|---|---|---|
| ラダー最高 | greedy+metal **532.8** | 提出履歴 |
| NashConv: net_run7\|metal | 0.13 | `results/nashconv.json` |
| NashConv: net_run7\|run7_best | 0.80 | 〃 |
| greedy地力: metal / run7_best / grass | 0.658 / 0.475 / 0.458 | `results/deck_strength.json` |
| go-first率: co-evoネット | 0.00（全部） | `results/opening_diag.json` |
| デッキ構成: trainer/energy | 17-19 / 27-31（異常） | notebook §2 |

**現状の暫定ベスト＝ `net_run7 + metal`**（我々の最頑健戦略, NashConv 0.13）。
未提出なので、**まずこれをラダー提出して“地力の基準点”を確定**するのが Step 0 的タスク。

---

## 3. 段階プラン（ステップごとに ablation）

各ステップ共通の進め方：
**(a) 1つだけ変える → (b) 短いQD/co-evoを1本回す → (c) §1スイートで前ベースラインと比較 →
(d) 勝ち候補をラダー提出 → (e) keep/drop を採否台帳に記録**。

実装は既存資産を流用：`scripts/qd_deck_search.py`（`_fitness`/`_evaluate`/`_build_seeds`）,
`src/qd/deck_qd.py`（`mutate`/`behaviour_descriptor`）, `src/qd/archive.py`,
`src/net/deck_factored.py`（サロゲートの特徴量）, `scripts/nashconv_eval.py` / `deck_strength.py`。

---

### Step 0（前提）— 基準点の確定＋held-out プール
- **やること**：(1) `net_run7 + metal` をラダー提出し基準点を取る。(2) §1.4 の held-out プール＋
  `heldout_eval.py` を作る。(3) `results/ladder_corr.csv` を新設し既存提出を埋める。
- **完了条件**：ラダー基準点が出て、held-out スイートが回る。
- **手間**：小。**リスク**：低。

---

### Step 1 — 辞書式適応度 × 複数パイロット（最優先・根本原因）
- **仮説**：自己参照の勝率最大化をやめ、「①勝率を主＋②相手間/パイロット間の勝率分散を辞書式の副」かつ
  「複数パイロット評価」にすれば、co-adapt が止まり**頑健で（誰が回しても強い）デッキ**が出る。
- **変更点**（`scripts/qd_deck_search.py`）：
  - `_fitness`：候補を **{greedy, 現net,(任意で heuristic)} の複数パイロット**で評価。
  - 適応度を**辞書式**に：主＝平均勝率、副＝相手×パイロット間の**分散（小さいほど良い）**。
    実装は「勝率を ε(=0.02〜0.03) 幅で同点丸め → 同点内は分散で順位」。`MapElitesArchive.insert`
    の比較キーを `(round(winrate/ε), -variance)` のタプルにするのが最小変更。
  - 色ペナルティ等の既存ソフト項は併存可（要A/B）。
- **評価**：出力アーカイブのベストセル群の **NashConv**（vs 固定母集団）と **held-out勝率** を、
  現行 run7 アーカイブと比較。go-first率も確認（副作用チェック）。
- **採否条件**：NashConv が現行ベスト（net系の最良 0.13 系）以下に下がる or held-out勝率が上がる、
  かつラダーA/Bで greedy+metal に近づく/超える。
- **手間**：小〜中（_fitness/比較キーの数十行）。**リスク**：分散項が強すぎると凡デッキ化 → εと
  「副はタイブレークのみ」を厳守。

---

### Step 2 — オンライン深層サロゲート（DSA-ME 型, 効率の核）
- **仮説**：各評価がフル対戦で高コスト。デッキ→勝率を予測する小ネットで**事前選別**し、有望児だけ
  実対戦すれば、同じ計算で**桁違いに多く・広く**探索でき、強いビルドに届く。
- **変更点**：
  - `src/net/deck_factored.py` の特徴量（＋必要なら value head）を入力に、**勝率回帰サロゲート**を新設
    （`src/qd/surrogate.py`）。
  - QD ループ：各世代で子を**多めに生成→サロゲートで上位K個に絞る→実対戦**。実対戦結果を
    サロゲートの教師に**逐次追加して再学習**（co-improving）。古典版の代替として SAIL（GP+UCBの
    acquisition map）も選択肢。
- **評価**：**同じ実対戦予算**で、Step1単体 vs Step1+サロゲート の到達ベスト NashConv/held-out を比較
  （効率＝同予算でどれだけ良いか）。サロゲートの予測 vs 実測の較正も記録。
- **採否条件**：同予算で到達品質が明確に向上（CI非重複）。
- **手間**：中〜大。**リスク**：サロゲートのバイアスで探索が痩せる → 探索領域のデータで逐次再学習＋
  たまにランダム実評価で較正。

---

### Step 3 — ヒューリスティック / active-gene 変異
- **仮説**：一様ランダム補充が無駄と構造的到達不能の元凶。人間的編集（似カード差し替え・
  パッケージ単位の足し引き・エネ枚数ブロック調整）にすれば、**まともなデッキ率↑**＋一貫性エンジン型に
  到達でき、効率と多様性が両立。空間は縛らない（どのデッキも依然到達可能）。
- **変更点**（`src/qd/deck_qd.py` `mutate`）：
  - カードを役割でクラスタ化（アタッカー/ドロー・サーチ/エネ/テック; `src/net/deck_factored.py` 流用）。
  - 変異を「同役割内の差し替え／パッケージ単位の足し引き／エネ枚数調整」に重み付け。active-gene＝
    盤面で機能する枠を優先。**自由swapも低確率で残す**（探索を殺さない）。
  - 交叉（セル間で役割パッケージを組み替え）を追加可。
- **評価**：同予算で「まともなデッキ率（合法かつ非負fitness）」「到達ベスト品質」「カバレッジ」を
  Step2ベースラインと比較。
- **採否条件**：効率（同予算品質）かカバレッジが向上、品質が悪化しない。
- **手間**：中。**リスク**：操作が賢すぎてメタ近傍に偏る → 自由swapの確率を確保しカバレッジで監視。

---

### Step 4 — coevolution（GAME型）＋ warm-start
- **仮説**：固定相手をやめ相手も進化させればバーが上がり続け、真に強いデッキが出る。アーカイブ持ち越しで
  改善が複利化。
- **変更点**（`scripts/train_qd_coevo.py`）：
  - 相手プールを**進化させる**（GAME型：両側MAP-Elites／交互に片側を“相手”として固定）。
  - **hall-of-fame**（過去の強相手）を保持して**循環**を抑制。
  - アーカイブを**ラウンド跨ぎで warm-start**（作り直しをやめる）。
  - 多様性低下に備え、**片側の一様アーカイブを多様性担保として併走**（ハイブリッド）。
- **評価**：NashConv（held-out母集団）と held-out勝率で、Step3ベースラインと比較。多様性
  （カバレッジ/セル間分散）も併記。
- **採否条件**：品質（NashConv/held-out）が向上し、多様性が許容範囲で維持。
- **手間**：大。**リスク**：循環・多様性崩壊 → HoF＋ハイブリッド＋カバレッジ監視。

---

### 補助つまみ — CMA-MAE の α（Step2 以降で emitter 化する場合）
- ランダム変異の代わりに CMA 系 emitter を使うなら、**α（soft archive 学習率）**で「強さ重視⇄多様性重視」を
  連続調整。α=0 純最適化 / α=1 純QD。pyribs 参照。Step2/3 を CMA-ME 化する際の選択肢。

---

## 4. 進め方（運用）
- **1ステップ＝1セッション**を基本に、本書の該当 Step だけを実装→評価→ラダー→採否台帳追記。
- 採否は PLAN.md の「採否台帳」に1行で記録（変更・ローカル差分・ラダー差分・keep/drop）。
- ラダーは5/日。提出は `dangerouslyDisableSandbox` 必須・非サンドボックスで（`~/.kaggle/credentials.json`）。
- コミットは作業ブランチ。各ステップ前に `uv run ruff check . && uv run ty check && uv run pytest -q`。

## 5. 主要参考文献
- García-Sánchez et al. 2016 (IEEE CIG) / 2018 (Knowledge-Based Systems) — 辞書式適応度・分散項・
  play/deck entanglement 明言・human-edit 変異。
- Miernik & Kowalski 2021 (arXiv:2105.01115) — 固定弱相手は退化、self-play優位、active-gene。
- Fontaine et al. 2019 (MESB, GECCO) — Hearthstone MAP-Elites、記述子＝マナ平均/分散。
- Zhang/Fontaine et al. 2022 (DSA-ME, GECCO) — オンライン深層サロゲートQD（SOTA）。
- Gaier et al. 2018 (SAIL, Evolutionary Computation) — GP+UCB acquisition map。
- Fontaine & Nikolaidis 2023 (CMA-MAE, RA-L) — α で最適化⇄多様性。
- García-Sánchez et al. 2024 (competitive coevolution) / GAME 2025 (arXiv:2505.06617) — coevolutionary QD。
- 評価methodology: Balduzzi et al. 2018 (Re-evaluating Evaluation, Nash averaging),
  Agarwal et al. 2021 (rliable), AlphaStar (league/exploiters), Procgen (held-out)。
