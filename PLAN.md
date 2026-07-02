# PLAN — Pokémon TCG AI Battle 攻略プラン

調査(`docs/research/game-ai-survey.md`)を踏まえた段階プラン。
**終着点(収束先)**: 添付論文
[Mastering Strategy Card Game (Hearthstone) with Improved Techniques (Xiao et al. 2023, arXiv:2303.05197)](https://arxiv.org/abs/2303.05197)
の手法（詳細まとめ→ **[docs/research/osfp-cardgame-2303.05197.md](docs/research/osfp-cardgame-2303.05197.md)**）——
**OSFP（Optimistic Smooth Fictitious Play）の自己対戦で学習した end-to-end 方策+価値ネット**を中核に据える。
論文どおり**デッキ構築も対戦と同時に学習する**（デッキ構築ヘッド=CBヘッド）。デッキは**手で固定せず**、学習された分布（混合戦略）からサンプルする。
ただし本コンペ制約(CPU・10分/試合・手番制限)に合わせ、論文が捨てた**探索を“推論時の上乗せ”として被せる**ハイブリッドに拡張する。
全工程を通して **ablation（1要素ずつ足し引きして効果を測り、使う/使わないを切り分ける）** を徹底する。

---

## 設計の中心思想（添付論文との対応）

> 論文の詳細は **[docs/research/osfp-cardgame-2303.05197.md](docs/research/osfp-cardgame-2303.05197.md)** に集約（OSFPの定義・Algorithm 1・improved techniques・本コンペとの差異）。以下はその要点対応表。

| 論文(2303.05197) | 本プランでの採り方 |
|---|---|
| end-to-end **方策+価値ネット**（共有埋め込み・再帰・価値ヘッド） | **中核**として採用。推論はforward pass主体＝CPU/時間制約に最適 |
| **デッキ構築ヘッド(CB)** をプレイ(BT)と共有埋め込みで同時学習 | **採用**。init(`obs.select is None`)で**合法マスク付き**に60枚を逐次生成。デッキを固定しない |
| **OSFP** 自己対戦（過去チェックポイント混合への smooth best response, 直近重み大→**last-iterate収束**） | **学習の中核**。最終チェックポイントをそのまま提出（平均方策の保持不要） |
| **autoregressive 行動分解** `(type,target)`＋0/1マスク（各stepの行動数を数十に） | Kaggleの `obs -> list[int]`(option index) 契約にそのまま対応させる |
| 隠れ情報は**決定化せず**観測履歴+再帰で暗黙学習 | 第一義はこれを踏襲。決定化は**推論時探索を入れる時のみ**登場 |
| **学習時も推論時も探索なし** | 学習は探索なしを踏襲（効率＋last-iterate維持）。**推論時のみ**任意で探索を上乗せ |
| 分散 actor-learner + V-Trace/PPO, 24GPU×10-23日 | GPU複数台/クラウド予算で**スケールダウン版**を回す。BCで暖機して収束を前倒し |

**探索の位置づけ（重要）**: 探索(PIMC/ISMCTS)は“主役”ではなく、学習ネットの上に被せる**test-time amplifier**。
学習ネットの**方策を prior（手順序付け）**、**価値をリーフ評価**に使う AlphaZe\* 風。
さらに探索/ヒューリスティックは **(a) 自己対戦の初期相手プール / (b) BC(行動クローン)の教師 / (c) 学習が間に合わない時のフォールバック** という支援役も担う。

---

## 基本原則（全Phase共通）

### A. ablation の規律
- **一度に変える変数は1つ**。エージェント比較時はデッキ固定、デッキ比較時はエージェント固定。
- すべての手法は **「前段ベースラインに対する勝率」** で評価し、**有意差が出たものだけ採用**する。
- 各実験は **keep / drop / 保留** を必ず判定し、§ 採否台帳 に記録する（なぜ採用/不採用かも）。
- 「効きそう」「論文で強い」は理由にしない。**この環境で測って勝ったものだけ残す**。
- **学習はループに探索を入れない**（論文の効率＆last-iterate収束を守る）。探索は推論時に分離して載せ、その効果を単独でablationする。

### B. 評価方法論（measurement）
- 比較は **同一シード・同一デッキ・先後を入れ替えて各 N 戦**（初期値 **N≥400**、僅差判定は 1000+）。
- 指標は **勝率 + Wilson 95%信頼区間**。CI が 50% を跨がない＝有意。僅差は試合数を増やす。
- **分散低減**: 可能な限り同一決定化シード/同一対戦カードで paired 比較。
- 採用の足切り（初期値、運用しながら調整）:
  - **明確採用**: 勝率 ≥ 55%（CIが50%超）
  - **保留/再計測**: 51–55%（試合数増やして判断、計算コストと相談）
  - **不採用**: ≤ 51% もしくは CI が 50% を含む
- 学習版は**対 固定相手プール（random/greedy/heuristic＋過去チェックポイント）の総合勝率**でも追跡し、過学習(特定相手特化)を検知。

### C. ローカル sim と実ラダーの二層評価
- ローカル sim は **高速な相対比較**用。ただし **実ラダー結果を外す**ことが知られる → 最終判断は実ラダー。
- **大きな意思決定（デッキ分布/プール方針・探索採否・学習版採否・最終提出）は必ずラダーで検証**（5サブ/日の枠を割く）。学習デッキ分布の価値（混合 vs 決定的）は特にローカルがラダーを外しやすい。
- ローカル順位とラダー順位の **ズレを記録**し、ローカル評価のバイアスを把握する。

### D. 提出衛生（毎Phase維持）
- **絶対にクラッシュしない**（例外時は合法手フォールバック）／**手番時間 ＆ 10分/試合を超えない**。
- 提出は `main.py`(トップ階層) + `deck.csv` + `cg/`（＋学習後はネット重み）の自己完結バンドル（ネット無し・CPU前提）。
- どのPhaseの版でも「いつでも提出できる動く状態」を保つ（退行したらすぐ気づける）。
- 学習ネットを載せる版では **推論ライブラリ依存を最小化**（重みを numpy/軽量ランタイムで読めるよう設計、CPU推論が時間予算内）。

---

## Phase 0 — 評価基盤とベースライン ✅(完了)
**目的**: 以降すべての ablation を同じ土俵で測れる、再現可能な対戦評価環境を作る。

**作るもの**
- Docker(linux/amd64) 上の **対戦ランナー**: 2エージェント×Nゲーム、先後入替、シード固定、勝率+CI出力。
- **エージェント登録**の仕組み（`agents/` に差し替え可能な方策を置く）。
- ベースライン2種: `random`（合法手ランダム）/ `greedy`（1手の即時報酬最大、例: 取れるKOを取る）。
- 結果ログ（CSV/JSON）と簡易集計（勝率・CI・平均ターン・時間/手）。

**達成基準（Exit）**: いずれも達成済み（§採否台帳・較正メモ参照）。
- random vs random ≈ 50%（±CI内）＝ハーネスが公平。
- `greedy` が `random` を **有意に**上回る（0.908, 500戦）。
- 同一シードで統計的に再現（engine RNGは非seedable→bit再現は不可）。1手/1試合の所要時間を計測済み。

---

## Phase 1 — デッキ空間の足場づくり（デッキは固定せず“学習対象”にする）
**目的**: デッキは最大のレバー。これを**手で1個に固定せず、CBヘッドで学習する**ための土台を作る。論文どおりデッキ構築を方策に内包する方針（→P3でヘッド化, P4で暖機, P5で対戦と同時学習）。

**作るもの**
- **合法カードプール/規則の特定**: エンジンが受理するフォーマット・禁止・枚数規則を `battle_start` の `errorPlayer`/`errorType` で**プローブして確定**（60枚ちょうど・同名4枚上限・ACE SPEC≤1・基本ポケモン≥1 等）。学習で扱うプールを定義（広めから／必要なら Standard 部分集合に絞る）。
- **合法マスク付きデッキ生成器/エンコーダ**: 60枚を逐次選択する各ステップに**合法マスク**（4枚上限・ACE SPEC・基本必須・残り枚数）。**常に合法デッキしか出せない**ことを保証（提出衛生§D／非合法デッキは即敗北のため必須）。
- **デッキ・デモンストレーション収集**: メタ/コミュニティのデッキリストを *最終選択肢ではなく* **BCの教師(prior)** として収集（P4でCBヘッドの暖機に使う）。
- **デッキ評価ハーネス**: エージェント固定でのデッキ総当たり。目的は「1個に絞る」ことではなく **(a)学習デッキ分布の健全性チェック / (b)自己対戦の多様な相手プール / (c)ローカル↔ラダー較正** の測定具。

**達成基準（Exit）**
- 合法/非合法の規則を把握し、**常に合法な60枚を生成できる**マスク実装が動く（ランダム生成でも `battle_start` がエラーを返さない）。
- 既知メタデッキ群の相性表が取れ、**デッキ差がエージェント差より勝率を動かす**ことを定量確認（レバーの裏取り）。
- 代表デッキで**初回ラダー提出**し、ローカル↔ラダー較正を開始（学習デッキ分布の最終検証はP5/P7）。

**Ablation**: エージェント固定でデッキだけ変え、**どの軸（アーキタイプ/エネ構成/技範囲）が勝率を動かすか**を把握＝CBヘッドが学ぶべき構造の事前知識にする。

⚠️ **重要な含意（init は無情報）**: init選択時は `obs.current`/`obs.select` が共に None＝**相手情報ゼロ**。よってデッキは相手に条件付けできない＝**無条件の分布（混合戦略）**になる。これは弱点ではなく、**多相手ラダーで付け込まれにくい混合デッキ戦略**を OSFP が学ぶ狙いと合致する（単一固定デッキはカウンターに弱い）。

---

## Phase 2 — ヒューリスティック評価関数（探索なし）
**目的**: 盤面/行動を点数化する評価関数を作り、**探索なしで** greedy を明確に上回る。学習・探索の“土台/教師/相手”を作る。

**作るもの**
- 状態評価の特徴量: prize差・場の総HP/期待打点・エネ付与状況・KO可否/被KOリスク・ベンチ展開・手札リソース等。
- 行動スコアリング（1手 or 浅い貪欲）。各特徴を **on/off 切り替え可能**に実装（ablation前提の設計）。

**達成基準（Exit）**: §Phase 2 メモ参照（サンプルデッキ・ミラーでは greedy と互角＝**保留**。デッキ確定後に再計測）。

**Ablation（特徴量レベル）**: 各特徴を1つずつ抜いて勝率変化を測り、**効く特徴だけ残す**。

**役割の再定義（収束方針に伴う）**: heuristic はこの先、
**(1) 強い固定相手ベースライン**、**(2) OSFP自己対戦の初期相手プール**、**(3) BC(Phase4)の教師方策**、
**(4) 推論時探索のリーフ評価/特徴のフォールバック** として再利用する（成果物は捨てない）。

---

## Phase 3 — 状態/行動表現 & ネット骨格（学習の前提インフラ）✅(完了)
**目的**: 論文の **end-to-end 方策+価値ネット**を載せるための、観測エンコード・行動分解・推論基盤を作る。ここは“配線”であって、まだ強さは問わない。

**作るもの**
- **観測エンコード**: 場/手札/サイド/エネ/相手公開情報を固定長 or 集合エンコード（カード埋め込み）。論文Table相当の特徴を網羅。
- **デッキ構築ヘッド(CB)**: init(`obs.select is None`)で**60枚を逐次・合法マスク付き**に出力（P1のプール/マスクを使用）。カード埋め込みは対戦ヘッドと共有。initは無情報入力なので「デッキ分布を生むだけ」のコンパクト経路。
- **対戦ヘッド(BT)の autoregressive 行動分解**: `(type, target)` を逐次選択＋**0/1合法手マスク**。Kaggleの `obs -> list[int]`(option index) 契約と**双方向に往復**できる薄いアダプタ。
- **価値ヘッド**（CB/BTで共有埋め込み、必要なら再帰/履歴で隠れ情報を吸収）。
- **CPU推論パス**: 重み保存形式＋軽量ランタイム（依存最小）。1手の推論時間を計測。

**達成基準（Exit）**: すべて達成（§Phase 3 メモ・採否台帳参照）。
- ランダム初期化ネットが**全局面で合法な forward pass**を返し、option-index契約を往復できる（クラッシュ0）。→ 実機 probe で crashes=0 / illegal=0。
- CBヘッドが**常に合法な60枚デッキ**を生成できる（`battle_start` がエラーを返さない）。→ greedy+sampled 全て errorType=0 受理。
- 1手あたりCPU推論が**手番予算に対し十分なマージン**（探索を被せる余地を残す）。→ avg ~0.1ms / 最悪 ~6ms。
- 軽い健全性: ネットが random/greedy の手をある程度模倣できる（学習配線が機能している確認）。→ numpy SGD で policy loss 低下・精度>chance。

**Ablation**: 入力特徴セット・埋め込み次元・再帰の有無を後段(Phase4/5)で振れるよう、**全て構成可能**にしておく（`NetConfig` で hidden/各ヘッド幅、固定特徴の各ブロックを差し替え可能に実装）。

---

## Phase 4 — 行動クローン(BC)で暖機（自己対戦の前倒し）✅(完了, CBの決定的生成のみ制約あり)
**目的**: スクラッチRLのコールドスタート（激重）を避けるため、**ヒューリスティック/探索を模倣**して非ランダムな初期方策＋価値を得る。OSFPの初期化に使う。

**作るもの**
- 教師の対戦ログ生成: Phase2 `heuristic`（および必要なら短い決定化探索）の (観測, 取った手) ペアを大量収集。
- **対戦ヘッド(BT)のBC**＋**価値ヘッドの回帰**（試合結果/割引リターンを教師に）。
- **デッキ構築ヘッド(CB)のBC**: P1で集めた**デッキリスト・デモ**を教師に、合法マスク下でCBヘッドを暖機。これは「デッキ固定」ではなく**学習の初期化**（OSFPで周辺を探索する起点）。

**達成基準（Exit）**: §Phase 4 メモ・採否台帳参照。BT/価値は全達成、CBは合法だが決定的生成に制約。
- BCネットが **`random` に圧勝**し、教師（heuristic/greedy）に**同等以上**まで到達。→ **達成**（vs random 0.998 / vs greedy 0.532 / vs heuristic **0.604＝教師超え**, 各500局・実機）。
- CBヘッドが**合法かつ既知メタ相当のデッキ分布**を生成できる。→ **部分達成**（常に合法・errorType=0。**sampled** decode は distinct=50 で多様だが、**greedy** decode は固定特徴ゆえ distinct=2 に崩壊＝個体選択は学習カード埋め込み待ち）。
- 価値ヘッドが試合結果を**ベースライン超**で予測（勝敗判別が偶然超）。→ **達成**（符号正解率 0.895 ≫ 0.5）。
- このBC版をOSFPの初期重みとして確定（＝Phase5のスタート地点）。→ **確定**（`data/bc/bc_net.npz`, heuristic-init）。

**Ablation**: 教師(heuristic vs 探索)・データ量・価値教師(最終結果 vs 割引リターン)、**CBの初期化(デッキBC vs スクラッチ)**を振り、**OSFP初期化として最良の構成**を選ぶ。
BCだけで既存ベースラインを超えるなら、それ自体を一つの提出候補として確保（保険）。

---

## Phase 5 — OSFP 自己対戦学習（中核：論文手法の本体／デッキ↔プレイ同時学習）🚧(5d 統合ループ実装・GPU/速度整備済, 本RLラン待ち)

> **進め方（更新）**: 当初は「1能力ずつ(5a→5b→5c)」だったが、5a(プレイRL)が天井・5b(文脈自由CB)が破綻・5c(LSTMデッキ頭)が固定デッキに勝てず、ユーザ指摘で**「固定デッキ前提を捨て、デッキ↔プレイを自己対戦で同時学習する」論文どおりの形へ収束**。**5d = 統合 joint OSFP**（πBT+πCB+**共有カード埋め込み**を1更新で同時学習）に各アームを畳んだ（下記 Phase 5d メモ）。器(LSTMデッキ頭・共有埋め込み・parity・OpponentPool)は 5a–5c の成果を再利用。
> **5a = デッキOFFアーム**（プレイ+価値だけRL・デッキ固定）= 天井 ~vs BC 0.56 で収束（台帳 6/21）。
> **5b = デッキONアーム（文脈自由CB）** = 全敗ゼロ信号＋構成崩壊で**破綻**→自己対戦スコアリング(5d)へ。
> **5c = LSTMデッキ頭＋学習埋め込み** = 器は完成・parity一致。だが固定デッキ超え不可。**5d で本領**。
> **5d = 統合 joint OSFP（現在地）** = 5a–5c を統合。CB重みのtype-target化＋構成制約greedyでデッキ品質を是正、GPU/サブサンプリング/obsコピー除去で **1iter 726→50s**。本格RLランはこの上で。
**目的**: 添付論文の中核を実装（アルゴリズム詳細→ **[docs/research/osfp-cardgame-2303.05197.md](docs/research/osfp-cardgame-2303.05197.md)** §3・§5・§6）。**過去チェックポイント混合への smooth best response** を分散自己対戦で学習し、**last-iterate収束**した最終ネットを得る。**デッキ構築(CB)と対戦(BT)を同時に学習**する。**学習ループに探索は入れない**。

**作るもの**
- **分散 actor-learner**（Docker sim を多数並列でデータ生成、GPUで学習）。off-policy補正は **V-Trace（＋PPO損失）**。
- **デッキ↔プレイ同時学習**: 各局の頭でCBヘッドが**デッキをサンプル**→そのデッキを `battle_start` に渡して対戦→勝敗が**デッキ選択にも逆伝播**（デッキの勝率寄与が学習信号）。自己対戦の相手も学習中のデッキを使うため、**相手デッキが自然に多様化**する。
- **OSFPメタループ**: 相手履歴 H を保持し、勝敗記録で重み付けサンプリング＋確率pで自己対戦。直近モデルに高重み（last-iterate）。
  - 初期相手プール = `random`/`greedy`/`heuristic`＋**P1のメタデッキ群**（Phase1/2の再利用）。以降は自分の過去チェックポイント（=その時点のデッキ分布込み）を追加。
  - 新チェックポイントを H に追加する条件（全履歴に対し閾値ξ超 or c周期）を実装。
- 学習安定化(論文の improved techniques から取捨): 割引γ・V-Traceクリップ・生産/消費バランス(actor数調整) 等を**1つずつablation**。

**達成基準（Exit）**
- OSFPネットが **BC版 と heuristic を有意に上回る**（≥55%, ローカル＆**ラダー**両方）。
- **対 固定相手プール（多様なメタデッキ）の総合勝率**がBC比で改善（exploitability低減＝付け込まれにくさの代理指標）。
- 学習された**デッキ分布が特定カウンターに過剰に弱くない**（混合戦略のロバスト性）。
- 最終チェックポイントを**そのまま提出**して退行なし。CPU推論が時間予算内。

**Ablation**: OSFP有無(=BC固定相手 vs 動的混合)、相手混合の重み(直近偏重 on/off)、**デッキヘッド学習 on/off（BCデッキ固定 vs 学習デッキ分布）**、**デッキの混合度（決定的ベスト1 vs 確率混合）**、improved techniques を各1要素。
**KEEP条件: ローカル＆ラダーで前段(BC/heuristic)を有意に上回る**こと。伸びが出なければ improved techniques の追加 or デッキ表現/プールに投資を戻す。

---

## Phase 5c — 観測履歴の再帰集約（LSTM）でネット容量拡張（隠れ情報を暗黙学習）

> **位置づけ（条件付きアーム）**: Phase 5a/5b は **memoryless ネット**（`encode_state` が「今の `current` だけ」を見る）で OSFP を回す。**それが頭打ち or さらなる伸びが欲しいとき**に、ネット容量を上げて論文の「隠れ情報は決定化せず**観測履歴＋再帰**で暗黙学習」を取り込むのがこの段。**§A の規律どおり「再帰を足して memoryless 版を有意に上回ったら採用、出なければ捨てる」**。学習カード埋め込み（Phase4 で判明した CB 個体識別の壁＝**5b の前提**）と同じ「ネットを重くする」系で、通常は **埋め込み→再帰** の順。安いレバー（チューニング・物量・x86 Linux 並列化）を使い切ってから着手するのが費用対効果的。

**目的**: 相手の手札・山・引きといった隠れ情報を、**観測の履歴を再帰（LSTM 等）で集約**して暗黙的に活かす。今は無視している Kaggle 観測の **`logs`（試合イベント履歴）** を入力に取り込む。論文は LSTM（隠れ256）で観測系列を畳み、決定化も明示的信念状態も使わずに勝敗(±1)から end-to-end 学習している（→ [docs/research/osfp-cardgame-2303.05197.md](docs/research/osfp-cardgame-2303.05197.md) §4）。

**作るもの**
- **履歴エンコーダ（再帰経路）**: 各意思決定の観測（＋`logs` のイベント差分）を時系列で食い、**局内で隠れ状態を持ち越す**再帰経路。学習側 torch に LSTM（隠れ次元は ablation）、**推論側は numpy で同等の再帰 forward**を実装（提出は numpy 維持。既存の torch↔numpy 橋渡し＝`to_numpy_net`/parity テストを**再帰状態対応に拡張**）。
- **ステートフルな `NetAgent`**: `act()` 呼び出しをまたいで隠れ状態を保持し、**局頭（`reset(seed)`）で初期化**。提出衛生（クラッシュ0・時間予算）と整合させる。
- **系列対応の学習パイプライン**: 今の独立 `(state, action)` サンプルを**系列（trajectory）単位**に拡張し、（truncated）BPTT で学習。`bc_data`/`build_policy_samples` と OSFP ループ（`train_osfp`）を**系列バッチ対応**に。
- **CPU 推論予算の再計測**: LSTM 化で1手推論が重くなるため、numpy 再帰 forward の avg/worst を計測し**10分/試合・手番制限に収まる**ことを確認（現状 avg 0.12ms と桁違いの余裕があるので中容量化の余地は大きい）。
- （隣接）**学習カード埋め込み**: 固定特徴→学習射影を学習可能化＝CB の個体選択を可能に（5b の前提）。再帰と同じ「容量拡張」軸なので、ここで一緒に ablation してよい。

**達成基準（Exit）**
- 再帰版が **memoryless 版を有意に上回る**（≥55%, ローカル＆**ラダー**両方）。出なければ **不採用**（軽い memoryless を残す）。
- torch↔numpy の**再帰 forward parity 一致**（既存 <1e-9 水準）＝「torch で学習→npz→numpy で serving」が安全に往復。
- **提出衛生維持**: クラッシュ0・**CPU 推論が時間予算内**（worst を計測）・init で必ず合法な60枚。

**Ablation（各1要素）**: 再帰 有/無、隠れ次元、履歴の長さ（truncation 窓）、`logs` 入力 有/無、（隣接）学習カード埋め込み 有/無。
**KEEP条件**: ローカル＆ラダーで memoryless 版を有意に上回ること。**隠れ情報の利得がこの環境で小さければ捨てて軽い版に戻す**（探索 P6 や蒸留 P7 に投資を回す）。

⚠️ **コスト注意**: numpy 再帰 forward ＋ parity 橋渡し・系列パイプライン（BPTT）・ステートフル agent はいずれも非自明な改修。**memoryless OSFP が頭打ちになってから**着手する（安いレバーを先に使い切る）。論文の 24GPU は主にこの種の大きいネット＋巨大自己対戦パイプラインを止めないための throughput 投資であり、容量拡張と GPU 投資は**セットで判断**する。

---

## Phase 6 — 探索の上乗せ（test-time amplifier, ablation で採否）
**目的**: 「論文の内容を探索も合わせて」の実体化。学習ネットを**学習し直さず**、推論時にだけ決定化浅探索を被せ、**勝率が時間予算内で上がるか**を検証する。

**作るもの**
- **決定化＋浅い探索**（PIMC→必要ならISMCTS）。隠れ情報を予測サンプル→短い探索→平均。
- 学習ネットの統合: **方策=prior（手順序付け/枝刈り）**、**価値=リーフ評価**。探索なし(純ネット)を常に対照に。
- **時間予算管理**（手番上限・10分/試合厳守、途中打ち切り、最悪ケースでも超過しない）。

**達成基準（Exit）**
- `net + 探索` が `純ネット` を **有意に上回る かつ 時間予算内**（ローカル＆ラダー）。
- **決定化数 × 探索深さ** スイープで knee(頭打ち点)を把握＝コスト対効果が分かる。

**Ablation（核心の切り分け）**: 探索 ON/OFF、決定化数{1,5,20,…}、深さ{1,2,3,…}、ISMCTS vs PIMC を等予算で。
- 有意に勝つ → **採用**（純ネットに探索を被せて提出）。
- 互角/僅差（調査が高分岐ゲームで警告する結果）→ **純ネットに戻す**（軽い方を残す）。
⚠️ 調査の警告: 分岐因子が大きいと探索が決定化UCTと**互角**になりうる → 「探索は当然強い」と仮定せず**実測で採否**。

---

## Phase 7 — 軽量化・頑健化・最終提出
**目的**: 制限時間に余裕を持って収め、**過剰適合を避けた頑健な最終版**を確定する。

**作るもの**
- 必要なら **方策蒸留**: 重い構成（net+探索 や 大きいネット）の出力を**軽量純ネット**に模倣学習で写し、推論を高速化。
- **デッキ分布のラダー検証**: 学習デッキ分布 vs その最頻デッキ(=決定的版) をラダーで比較（ローカルは実ラダーを外すため、混合の価値は必ずラダーで確認）。
- 時間予算ハードニング（最悪ケースでも超過しない）、フォールバック網羅、リーグ的多相手評価。

**達成基準（Exit）**
- 手番時間/10分に**マージンを持って**収まる・**全局面でクラッシュしない**（initで**必ず合法な60枚**を返す）。
- **多様な相手**（自作の各版＋メタ対抗デッキ）に対して頑健、ラダーでも退行なし → 最終提出。

**Ablation**: **蒸留版 vs フル版**（蒸留で失う強さは許容内か）／**多相手ロバスト性**（単一相性への過剰適合チェック）／**デッキ混合 vs 決定的ベスト1**（ラダーでどちらが堅いか）。

> 安全網: CBヘッド不調・デッキ分布がラダーで負ける場合に備え、**学習分布の最頻デッキ（決定的版）を常備フォールバック**にする。これは「デッキ固定への後退」ではなく**学習した分布の縮約**（同じネットの引数違い）。

---

## 手法の採否仮説（事前の見立て・最終は実測で更新）

| 手法 | 事前見立て | Phaseで判定 |
|---|---|---|
| **デッキ構築を学習(CBヘッド・混合戦略)** | **使う**（最大レバー。固定せず学習＝論文準拠） | P1,P3,P4,P5 |
| デッキの合法マスク＋プール定義 | **使う**（必須。非合法デッキは即敗北） | P1,P3 |
| デッキBC（メタデッキリストで暖機） | **使う**（CBの初期化。スクラッチ回避） | P4 |
| ルールベース/合法手フォールバック | **使う**（必須・安全網） | P0, 全Phase |
| ヒューリスティック評価関数 | **使う**（学習の教師/相手/フォールバック土台） | P2 |
| 状態/行動表現＋方策価値ネット骨格 | **使う**（学習の前提インフラ） | P3 |
| 行動クローン(BC)で暖機 | **使う**（RFコールドスタート回避＋保険提出） | P4 |
| **OSFP 自己対戦（中核・論文本体）** | **使う**（終着点。GPU予算あり） | P5 |
| improved techniques（γ/V-Trace/生産消費バランス等） | **1要素ずつ計測して取捨** | P5 |
| **観測履歴の再帰(LSTM)** で隠れ情報を暗黙学習 | **計測して判断**（memoryless版を有意超なら採用・容量拡張） | P5c |
| 学習カード埋め込み（固定特徴→学習射影） | **計測して判断**（CB個体選択=5bの前提／容量拡張） | P5b,P5c |
| 決定化+探索(PIMC/ISMCTS) を**推論時に上乗せ** | **計測して判断**（時間予算内で純ネット超なら採用） | P6 |
| 方策prior / 手順序付け（探索内） | **計測して判断** | P6 |
| 方策蒸留(distillation) | **計測して判断**（軽量化の手段） | P7 |
| **学習ループ内の探索**(AlphaZero型MCTS自己対戦) | **使わない寄り**（論文の効率/last-iterateを壊す。推論時のみに分離） | — |
| フル MuZero 系を自前学習 | **使わない寄り**（GPU/学習基盤が重すぎ） | — |
| CFR/Deep CFR/ReBeL のフル均衡 | **使わない寄り**（OSFPで均衡近似を代替。発想だけ借用） | — |
| DeepNash の R-NaD フル学習 | **使わない寄り**（学習大規模。OSFPで軽く代替） | — |
| LLM を推論ループ内で使用 | **使わない**（CPU/時間/ネット制約で非現実的。設計役はオフラインで活用） | — |

> 「使わない寄り」でも **アイデアは借用**する: 確率性→*Stochastic MuZeroのafterstate*、付け込まれにくさ→*均衡(OSFP/DeepNash/ReBeL)*。

---

## 採否台帳（実験ログ — 進めながら追記）

| 日付 | Phase | 実験(変えた1要素) | 比較対象 | 試合数 | 勝率±CI | ラダー | 判定 | メモ |
|---|---|---|---|---|---|---|---|---|
| 2026-06-19 | P0 | random vs random (seed 0) | — | 500 | 0.498 [0.454, 0.542] | — | 公平 | CIが50%跨ぎ＝ハーネスにスロット偏りなし |
| 2026-06-19 | P0 | random vs random (seed 500) | 上の独立再現 | 500 | 0.512 [0.468, 0.556] | — | 公平 | 別シードでも50%付近、較正OK |
| 2026-06-19 | P0 | **greedy** vs random | random ベースライン | 500 | **0.908 [0.879, 0.930]** | — | **採用** | 探索の土台＝greedyを基準ベースラインに |
| 2026-06-19 | P0 | greedy vs random (同設定再走) | 上の再現性確認 | 500 | 0.906 [0.877, 0.929] | — | 再現 | 0.2pp差。engine RNGは非seedable→統計的再現のみ |
| 2026-06-19 | P2 | heuristic vs **random**（健全性） | random | 500 | 0.916 [0.888, 0.937] | — | 健全 | greedy(0.908)と同等にrandomを圧倒＝壊れていない |
| 2026-06-19 | P2 | **heuristic** vs greedy (s0) | greedy ベースライン | 500 | 0.510 [0.466, 0.554] | — | **保留(不採用)** | CIが50%跨ぎ＝有意差なし。≥55%の足切り未達 |
| 2026-06-19 | P2 | heuristic vs greedy (s500) | 上の独立再現 | 500 | 0.530 [0.486, 0.573] | — | 保留 | 別シードでも50%付近、跨ぎ |
| 2026-06-19 | P2 | heuristic vs greedy (s1000) | 同上, 高N | 1000 | 0.496 [0.465, 0.527] | — | 保留 | 高Nでほぼ50%。サンプルデッキ・ミラーでは greedy と互角 |
| 2026-06-19 | P2 | ablation: drop **promote** | full(0.510) | 500 | 0.474 [0.431, 0.518] | — | promote=効く(寄与+3.6pp/dir) | 抜くと下がる→KO後の最強昇格は有効（ただしCI跨ぎ＝有意未満） |
| 2026-06-19 | P2 | ablation: drop **attach_target** | full(0.510) | 500 | 0.478 [0.435, 0.522] | — | attach_target=効く(+3.2pp/dir) | エネ的配分は有効（有意未満） |
| 2026-06-19 | P2 | ablation: drop **retreat** | full(0.510) | 500 | 0.532 [0.488, 0.575] | — | retreat=有害(-2.2pp/dir) | 抜くと上がる→アグロ・ミラーでは退却はエネ捨て＝テンポ損 |
| 2026-06-19 | P2 | drop retreat（新シード検証） | greedy | 1000×2 | 0.502 [.471,.533] / 0.531 [.500,.562] | — | 保留 | 新シードで~0.52。実在するが極小・有意未満 |
| 2026-06-19 | P2 | ablation: drop attack_ko / weakness / bench_dev | full(0.510) | 各500 | 0.504 / 0.506 / 0.512 | — | 中立 | 方策の選択を変えない（弱点は一様倍率で argmax 不変、攻撃は提示済み合法手の最大打点）。評価関数(探索/学習)用に保持 |
| 2026-06-20 | P1 | デッキ合法性プローブ（engine vs 自作validator, 6変種） | — | — | — | — | **一致確認** | `src/deck.py` の規則が engine と完全一致。errorType: 0=OK / 2=同名5枚 / 3=基本ポケ0 / 4=ACE SPEC×2。size≠60は wrapper の ValueError。全1267プールのランダム合法デッキも受理 |
| 2026-06-20 | P1 | デッキ評価ハーネス スモーク（sample+random合法3, greedy固定） | — | 60/対 | spread **0.794** | — | ハーネス健全 | サンプル周辺0.983, ランダム0.19–0.56＝デッキ差≫エージェント差(~0.50)。ランダムは極端例、メタデッキで本評価予定 |
| 2026-06-20 | P1 | **本デッキ評価**（デモ8 mono-aggro＋sample＋random, greedy固定） | — | 80/対(10デッキ) | spread **0.683** | — | コヒーレント確認 | 全デモがランダムを~1.00で圧倒（psychic_aggroのみ~0.45の弱外れ）。周辺: metal 0.744/sample 0.717…psychic 0.103。弱点相性構造あり。デッキ差≫エージェント差を**実デッキで**再確認 |
| 2026-06-20 | P1 | 提出バンドル スモーク（自己完結 greedy+metal_aggro, 自己対戦） | — | 1局 | — | — | 提出可 | `submission/main.py` が deck.csv読込＋all_attack()打点表(1556)＋98手完走・クラッシュ0。ラダー提出は要 Kaggle 認証（ユーザ操作） |
| 2026-06-20 | P1 | **初回ラダー提出**（greedy + metal_aggro, tar.gz/CLI, ref 53874111） | — | — | — | **≈602 / 1378位(2174)** | **M1起点** | 検証COMPLETE。516.6→602.6と上昇中＝レート変動継続中。ローカル↔ラダー較正の最初の点。提出手順/ハマり所は CLAUDE.md |
| 2026-06-20 | P3 | ネット骨格 forward 合法性（純numpy net、提示optionを採点、実機） | — | 6局(probe)+30局(eval) | — | — | **配線OK** | NetAgent が実機で **crashes=0 / illegal=0**（~550選択+30局）。option-index契約往復・value∈[-1,1]。random初期化なので強さは未（net 0/30 vs greedy=想定） |
| 2026-06-20 | P3 | CBヘッド デッキ生成（greedy+sampled×2, 合法マスク, 実機） | engine `battle_start` | 3デッキ | — | — | **常に合法** | 全て **errorType=0 受理**。distinct names: greedy 15(=4×15名)/sampled 59・56＝混合で多様化。論文「デッキは分布」の足場 |
| 2026-06-20 | P3 | 学習配線サニティ（**torch+Lightning** で policy BC, 合成データ） | — | — | — | — | **配線OK** | torch fwd/bwd/Adam が機能: loss 初期比<0.7・精度 chance(0.25)→>0.5。`Trainer.fit`→npz エクスポート→numpy serving が往復。torch↔numpy forward **parity<1e-9**。Phase4 BC の本体 |
| 2026-06-20 | P3 | 推論時間（net 1手, metal_aggro, 実機） | greedy(~0.002ms) | 30局 | — | — | **余裕大** | avg **0.12ms** / 最悪 **6.4ms**。greedy比~50倍だが sub-ms 中心＝探索/学習を載せる予算十分（手番制限に桁違いのマージン） |
| 2026-06-20 | P4 | **BC net(heuristic教師)** vs random（metal_aggro固定, 実機） | random | 500 | **0.998 [0.989, 1.000]** | — | **採用** | ランダム初期化net(0/30)から一変＝**random圧勝**。Exit「random圧勝」達成 |
| 2026-06-20 | P4 | BC net(heuristic) vs **greedy** | greedy | 500 | 0.532 [0.488, 0.575] | — | 同等 | CI跨ぎ＝greedyと**同等**（Exit「教師同等以上」を満たす） |
| 2026-06-20 | P4 | BC net(heuristic) vs **heuristic(教師)** | heuristic | 500 | **0.604 [0.560, 0.646]** | — | **教師超え** | CIが50%跨がず＝**教師をわずかに上回る**（BC平均化＝設計の論理pt4。最終的上積みはP5） |
| 2026-06-20 | P4 | ablation: **教師 greedy vs heuristic**（同一ログ流用） | heuristic-init各値 | 各500 | greedy-init 0.990/0.488/0.580 | — | heuristic優位 | 全対戦で heuristic-init ≥ greedy-init → **既定=heuristic 確定**（P5初期化） |
| 2026-06-20 | P4 | 価値ヘッド符号正解率（held-out, 23k samples） | chance 0.5 | — | **0.895** | — | **偶然超** | 実勝敗で学習＝真の信号。Exit「value偶然超」達成（policy top-1 acc 0.874 も同時） |
| 2026-06-20 | P4 | CB head decode（学習CB, 実機 battle_start） | — | — | — | — | **制約判明** | **greedy decode は distinct=2 に崩壊**（固定特徴＝プロファイル選択＋energyがcap免除）。**sampled decode は distinct=50 で合法**・多様。crash0/illegal0/worst0.87ms |
| 2026-06-20 | P5a | OSFP相手プール（recency/admission/self-play, 純粋） | — | — | — | — | **配線OK** | `src/net/osfp.py`：直近重み単調・baseline下限・self_play_prob・閾値/patience 採用を単体テスト（test_osfp 11ケース）。`cg`/torch非依存でネイティブ検証 |
| 2026-06-20 | P5a | LitPolicyGradient sanity（REINFORCE+baseline+entropy, 合成） | — | — | — | — | **配線OK** | +advで logp↑・value が returns に回帰・**masked entropy が NaN安全**（padding下）・**CBヘッド凍結**（trunk/policy/valueのみ更新）。`run_osfp` 全ループを fake generator でネイティブ検証（test_rl 9ケース）。デッキOFFアーム |
| 2026-06-20 | P5a | OSFP self-play collect＋loop（Docker `--smoke`, 実機） | — | 3iter×8局 | — | — | **配線OK** | `collect_selfplay`: 8局~1s・winner全decisive・学習者/相手タグ片側ずつ（リーク無）・learner単一選択303=サンプル。`train_osfp --smoke`: opp=random/self を pool から選択・**自己対戦は両スロットlearnerで~2倍サンプル**・iter3でpatience採用・final.npz出力（計6.3s） |
| 2026-06-20 | P5a | 学習後 net 実機 probe（final.npz, Docker） | — | 6局 | — | — | **PASS** | net vs greedy/net **crash0/illegal0**・worst **1.0ms**。**CBヘッド凍結を実証**（greedy distinct=2/sampled distinct=50＝Phase4と一致＝cb*未更新）。errorType=0 |
| 2026-06-20 | P5a | **OSFP net(100iter×256, default) final** vs **BC** | BC net | 500 | **0.546 [0.502, 0.589]** | — | **保留(lean)** | CI下限0.502＝BCを僅かに上回るが≥55%未達（§B 51–55%＝要N増/調整）。metal_aggro固定・argmax・~30分・33ckpt採用 |
| 2026-06-20 | P5a | OSFP net(同上) final vs **heuristic** | heuristic | 500 | **0.682 [0.640, 0.721]** | — | **改善** | BC(0.604)を~8pp上回る＝OSFPで強くなる方向は出た。ループ内eval(各100局)は~0.50/0.64でノイジー横ばい、**last-iterate(final)が最良**＝OSFPらしい |
| 2026-06-21 | P5a | **lr↓temp↓**（run2: lr1e-3→3e-4・temp1.0→0.5, 100iter） vs BC | BC net | 500 | 0.554 [0.510, 0.597] | — | **安定化(採用)** | run1 final と**互角**（vs run1 0.490 [0.446,0.534]）＝強さは不変だが、チェックポイント・トレンドの**振動が消え last-iterate が最良**に＝finalを提出可。安定設定として採用 |
| 2026-06-21 | P5a | **スケール14x**（run3: 同安定設定で 1400iter, 482ckpt） vs BC | BC net | 500 | 0.562 [0.518, 0.605] | — | **天井不変(不採用)** | run2 final と**互角**（vs run2 0.508 [0.464,0.552]）。**~iter350 で頭打ち→1000+iter横ばい**。vs heuristic も ~0.69 で run1/2 と同帯。**「もっと回す」では伸びない**ことを決定的に確認（次は容量/別ヘッド＝5b） |
| 2026-06-21 | P5a | **結論**: 5a は天井で収束（vs BC ~0.56・vs heuristic ~0.68） | — | — | — | — | **一区切り** | プレイ専用RL（デッキ固定）の上積みは~+8ppで頭打ち。瞬間崩壊は残存（PPO/V-Trace で対処可・天井は上げない）。ボトルネック=データ量でなく**容量/表現**。→ **5b（デッキ学習＝最大レバー）へ**。可視化: `notebooks/02_osfp_training.ipynb` |
| 2026-06-21 | P5b-i | **学習カード埋め込み**（固定特徴⊕embed16・CB BC再学習・実機probe） | Phase4 CB(distinct=2) | — | — | — | **崩壊解消(達成)** | greedy decode が **distinct 2→16**・engine **errorType=0**・demo重複0.48（sampled 30/29も合法）。crash0/illegal0/worst**0.79ms**・PASS。policy/value不変(0.872/0.893)＝CB独立。連結必須(デモ37種/プール1267)・parity非転置・旧npzは移行。`bc_net_emb.npz`(42,787 params) |
| 2026-06-21 | P5b-ii | **CBヘッド自己対戦RL**（infra＋スモーク＋ゲート, 実機） | 固定 metal_aggro | gate60 | gate **0.0** | — | **不採用(ゲート不達)** | パイプライン完走（収集→REINFORCE→ゲート）・native test緑(130)・`cb_pg_loss`/`cb_rl_samples`/`LitCBPolicyGradient`/`collect_cb`/`train_cb`。だが BC-CBデッキが metal_aggro に**全敗**。**全デッキ敗北→advantage≈0→REINFORCE信号ゼロ** |
| 2026-06-21 | P5b | **根本所見**: 文脈自由CB＋per-card decode はデッキ構成を守れない | — | — | — | — | **方針転換** | greedy=**エネルギー0枚**(43ポケ/17トレ/0エネ＝機能せず)・sampled=エネ6–14(metal_aggro 31)。エネ比率のような**大域制約は per-card context-free スコアで表現不可**（greedyがエネを押し出す/inverse-copy重みでも thread 不能）。→ **CB-RL不採用・固定デッキ維持**（§A・Phase7安全網）。デッキ学習には**autoregressive デッキ方策**（各picを部分デッキで条件付け）が必要＝将来課題 |
| 2026-06-22 | P5c-i | **LSTM自己回帰デッキヘッド**（記憶付き・部分デッキ条件付け, 実機probe） | 5b(0エネ崩壊) | — | — | — | **構成崩壊を解消(達成)** | greedy が **エネ35/ポケ8/重複0.80**（5bは0エネ）・errorType=0・crash0/worst2.46ms PASS。numpy LSTM forward と torch の **parity<1e-9**（非転置bridge・gate順i,f,g,o・H≠in test）。十分なCB BC(150ep/shuffle12/hidden64)が必要（40epでは46/1に偏る） |
| 2026-06-22 | P5c | ゲート: LSTM greedyデッキ vs 固定metal_aggro（同プレイ頭, 実機） | metal_aggro | 80 | **0.000** | — | **不採用(ゲート不達)** | 構成は均衡だが**型不整合**: Grassエネ35＋Zapdos(雷)/Glastrier＝**技が撃てない**・非ex弱攻撃役8枚。**8デモ横断BCは「汎用骨格」を学ぶがアーキタイプの型一貫性を学べない**→ coherent な metal_aggro に全敗 |
| 2026-06-22 | P5c | **結論**: LSTMデッキヘッドは正しい器・だがデッキ学習は固定デッキに勝てず | — | — | — | — | **固定デッキ維持** | 器（LSTM・parity・機能デッキ生成）は完成・再利用可。8デモBCで metal_aggro 超えは無理（型一貫性/データ量不足、RLゲートも信号薄）。§A→**固定 metal_aggro 維持**（Phase7安全網）。将来: 単一アーキタイプBC＋curriculum RL |
| 2026-06-22 | P5c→修正 | **方針転換**: 固定デッキはスケールしない／「vs固定で全敗=ゼロ信号」は設計ミス | — | — | — | — | **vs固定を撤去** | ユーザ指摘: 自己対戦なら試合は無限生成可＝「データ不足」は誤り。真因は **(a) スコアリングを“デッキ vs デッキ自己対戦”に直す ＋ (b) 計算速度**。固定デッキ前提を捨てる |
| 2026-06-22 | P5d | **デッキ自己対戦OSFP** 実装＋実機スモーク（相手デッキもCBサンプル＝自分/過去ckpt・同凍結プレイ頭） | — | 3iter×4×4 | self mean_wr **0.44–0.69** | — | **設計確立(信号あり)** | `collect_deck_selfplay`＋`train_deck_osfp`(OpponentPool＋LitCBSeqPolicyGradient)。相手=自分のデッキ分布→対称で約0.5・**勝敗ばらつき→advantage≠0＝信号常時あり**(5b-iiの全敗ゼロ信号を解消)。`collect_cb`/`train_cb`(vs固定)は**削除**。学習の伸びは長時間run=計算力(後で・クラウド) |
| 2026-06-22 | P5d-速度 | スループット実測（単一）＋コレクタK並列(Docker)実装・実測 | — | 512試合 | **9.6–11.6 試合/秒** | — | **本Macでは並列無効・コードは保持** | 単一: ~11.6試合/秒＝7hで**~25万試合**(論文3.2億の0.08%)。並列実測: x86エミュ(aarch64上x86_64・VM=15コア全使用可)が**並列で競合**→ w1=9.6 / w2=4.1 / w6=3.8 試合/秒＝**workers>1は逆に遅い**(1コンテナでエミュ飽和)。∴既定`--workers 1`。**>1はnative x86(クラウド/Linux)でのみ有効**＝真の高速化はエミュ撤廃(クラウド) |
| 2026-06-22 | P5d-native | **native x86機(16コア+A100)へ移行**・並列スケール実測 | — | — | — | — | **線形スケール(条件付き)** | 真因はエミュでなく **numpy OpenBLAS の過剰サブスクリプション**: 各コンテナが全コア分のBLASスレッドを起動し競合。**1スレ固定で workers≈線形**(64試合/コンテナ: 1→16並列で 4.9→57 試合/秒, ~12並列まで線形)。pin無しは8並列で各15x遅。`--native`(Docker層除去)で起動/デーモン競合も消えさらに速い。`train_*osfp`の`DOCKER_PREFIX`にスレ固定+ `--native`追加 |
| 2026-06-22 | P5d-engine | エンジンが現実型デッキ(低エネ+ドロー/サーチ)を扱えるか実機検証 | — | — | — | — | **完全サポート確認** | サポート61/グッズ77が本物のテキストで実装。**実際にプレイして効果発火を確認**(Billy=ドロー+条件分岐, Buddy-Buddy Poffin=デッキサーチでベンチ展開)。∴エンジン側に低エネ現実型の制約なし(=デッキ空間は本物のモダンポケカ) |
| 2026-06-22 | P5d-heuristic | TCGベストプラクティス(掘ってから貼る/need-awareサブ選択)を heuristic に反映 | full heuristic(mirror) | 各300 | dig_first **0.42**(lowE) / card_select 0.47 | — | **不採用(撤回)** | 教科書手順を入れても **neutral〜悪化**。1-ply heuristic は「掘る」判断が雑で行動数上限に達しエネ貼りを取り逃す=生産的に掘れない。§A→両方revert。**play質の向上は手書きルールでなくRL(πBT)が筋**(joint OSFPの価値の裏付け) |
| 2026-06-22 | P5d-CB重み | **CB BC の type-target 重み**(タイプ別総重み=物理比率, タイプ内は等重み) | inverse-copy(energy=3) | — | — | — | **エネ枯渇解消(採用)** | `1/copies`がエネを52%→学習重み11%に潰していた(=energy枯渇)のを修正。sampled デッキが **demo構成(31/12/17)に一致**。Phase4/5cの「greedy 0/46エネ崩壊」「型不整合」の根本側を是正。`cb_sequences`の`_type_target_weights` |
| 2026-06-22 | P5d-greedy | **構成制約付き greedy decode**(各タイプをネット自身のsampled平均でcap) | 無制約greedy(0/46エネ) | — | — | — | **機能デッキ化(採用)** | argmax が最頻カード(単エネ)を増幅する脆さを是正。greedy デッキ **energy=29/pokemon=15/distinct=10**(=[15,35]内・機能的)。sampled は無制約のままで RL の探索自由を維持。`build_deck`の`_type_caps`/`card_kind`(`src/deck.py`) |
| 2026-06-22 | P5d-joint | **joint OSFP 実装**(πBT+πCB+**共有カード埋め込み**を同時学習) | 旧 5a/5b/5c 分離ループ | 1iter | self mean_wr ~0.5 | — | **統合ループ確立** | 共有埋め込みを**playヘッドにも注入**(encode行関数・numpy/torch forward両方で lookup・parity<1e-9維持・任意サイズ load可)。`LitJointPolicyGradient`(凍結なし1更新, CombinedLoader)。旧 `train_osfp`/`train_deck_osfp`/2コレクタを**削除**し `train_joint_osfp`+`collect_joint_selfplay` に統合。BC再訓練で初期分布も健全化 |
| 2026-06-22 | P5d-GPU/速度 | GPU修復(torch cu124, **A100**)+学習GPU化+サブサンプリング+obsコピー除去 | 元(CPU・間引きなし) | — | — | — | **1iter 726→50s(14.5x)** | torch を cu124 ビルドに固定(Linux x86のみ; Mac非影響)。更新の GPU speedup **31x(小)/57x(0.44M大)**。play決定を~2万にサブサンプル(encode前にコレクタで間引き=deepcopy~9x減)。0.44Mネットを GPU **60s** で訓練・機能デッキ。**500iter=一晩/1000iter=~14h**(元8.4日) |
| 2026-06-22 | P5d-RL診断 | joint OSFP 本ラン(run1)のデッキ崩壊を多面調査 | — | — | — | **崩壊=真因** | デッキが **iter2 で E29/P15→E59/P1 に崩壊**し以降ずっと degenerate(1ポケ59エネ)。gateの「0.05→0.8→0.0」発振は**崩壊済みデッキに対するgateのノイズ**(同一ckpt再評価で0.66→0.517、エンジン乱数非seedable)。真因=(a)自己対戦のデッキ信号が弱(grounding欠如=両者一緒に崩壊→ミラー~0.5→advantage薄)+(b)デッキアームに正則化皆無+(c)`CombinedLoader`が deck loss を**76×サイクリング**適用。play頭は健全 |
| 2026-06-22 | P5d-修正 | デッキアームに **entropy+BC-KLアンカー** 追加＋76×除去(単一fit/`deck_batches`ガード) | 崩壊版(run1) | — | — | — | **崩壊解消(採用)** | 検証/本ラン(run2)で greedy/sampled が **E∈[28,32] を維持・崩壊せず**。run2 gate: **BC 0.175→iter25 0.55**(大改善)だが**改善はplay頭のみ**(デッキ構成はBCと完全同一=KL=1.0が強く凍結)。`LitJointPolicyGradient`に`deck_entropy_coef`/`deck_kl_coef`/`ref_net`(BC参照, KL(ref‖cur)=soft蒸留)。**デッキ自体も強くするには KL↓ or ルール制約**(下記メモ) |
| 2026-07-02 | デッキ探索Step3 | **heuristic/active-gene変異**(同役割差替・プレイセット単位抜差・エネ枚数調整＋低確率自由swap; `card_role`＋`--mutation` A/B, PR#18) | Step1一様swap(同一設定・seed0のみ差) | 20/100gen対照＋再評価1296局/deck＋held-out 10800局 | 再評価: heur 0.704±.022 / rand 0.720±.021(**互角**)。held-out(greedy操縦): heur 0.698 / rand 0.722 | — | **KEEP(オペレータ)** | 本旨=**停滞解消は達成**: admitted 47 vs 9(最終admit gen94 vs 66)・coverage 13 vs 11・アーカイブ平均 0.352 vs 0.337(終盤まで上昇)・トレーナー正常セル 9/13 vs 4/11。**到達デッキ質は互角**(エンジン乱数でbest一点値は±0.04ぶれ=20genの0.704 vs 0.668もノイズ)→「生命力→質」の転換は Step2(サロゲート)へ。co-adapt罠なし(greedy操縦0.70前後; run7_best 0.418と対照)。デッキ単体最強は依然 qd_step1_best(held-out 0.753)=ラダー候補。results/qd_step3_{heur,rand}{20,100}.json, heldout_qd_step3.json |

### 較正メモ（Phase 0 で確定した運用値）

- **エンジン乱数は seed 不能**: 公開API（`battle_start`）にシード引数が無く、同一エージェントseedでも局ごとに展開が変わる（実測: 同seedで52手 vs 15手）。→ **局単位のbit再現は不可能**。エージェント側乱数のみseedし、全結果をCSVに記録、**先後入替＋大N＋Wilson CI** で統計的に再現する運用とする（採否判定はこれで足りる）。
- **必要試合数の目安**（Wilson CI 半幅から逆算）:
  - **N=500**: 半幅 ≈ ±4.4pp。**55%級の差はギリ有意**（55%なら下限≈50.6%）。明確採用/不採用の一次判定はこれで十分。
  - **僅差（52–53%級）**: N=500では50%を跨ぐ → **N≈1,000–2,400** が必要（§B「僅差は1000+」を定量裏付け）。
  - 運用: まずN=500でスクリーニング、跨いだら増やす。
- **計測コスト**: Docker(linux/amd64エミュ)で **約180–340 games/sec**（greedy同士は短手数で速い）。500戦≈1.5–2.8秒。ローカルでの大量ablationは安価。
- **手番時間**: 両ベースラインとも1手 **平均~0.002ms / 最悪~0.2ms**。制限時間に対し桁違いに余裕（探索/ネット推論を載せる予算が十分にある）。
- **足切り閾値は §B の初期値のまま据え置き**（明確採用≥55% / 保留51–55% / 不採用≤51%）。N=500で運用可能と確認できたため変更不要。

### Phase 1 メモ（デッキ空間の足場 — 一巡完了）

- **作ったもの**: `src/deck.py` — デッキ構築規則の **validator（`legality_errors`/`is_legal`）** と、CBヘッド用の **逐次合法マスク（`legal_next_ids`）**＋**ランダム合法デッキ生成（`random_legal_deck`）**。`cg`・pandas 非依存の純関数（`build_pool` のみ card CSV を読む）でネイティブ単体テスト可（`tests/test_deck.py`, 16ケース）。
- **エンジン規則を確定（`scripts/probe_deck_legality.py`, Docker）**: 自作 validator が **engine と完全一致**。`StartData.errorType` の対応 = **0:OK / 2:同名5枚以上 / 3:基本ポケモン0 / 4:ACE SPEC 2枚以上**。`size≠60` は `battle_start` の `ValueError`（engine 到達前）。**4枚制限は名前単位**（基本エネは免除）。
- **プール所見**: 全 **1,267枚**。`Basic Pokémon` 1,056・`Trainer` 191・`Energy` 20、ACE SPEC 29枚。**154 の名前が複数 card_id にまたがる** → 4枚制限は ID でなく**名前で数える**実装にしてある。全プールからのランダム合法デッキは engine が受理＝**Standard 等の追加フォーマット制限は（少なくともこの範囲では）検出されず**。
- **デッキ評価ハーネス**: `scripts/run_deck_eval.py`（`run_eval` 再利用の総当たり＝同一ポリシー・デッキだけ変える）。勝率行列＋周辺ランキング（対 field 平均勝率）＋ spread を出力。スモーク（サンプル＋ランダム合法3, greedy固定, 60戦/対）でサンプルが圧勝＝周辺 **0.983** vs ランダム 0.19–0.56、**デッキ spread 0.794 ≫ エージェント差(~0.50 ミラー)**。「デッキ差＞エージェント差」の一次裏取り（※ランダムは極端例。メタデッキで本評価する）。
- **デモデッキ（BC教師の素体）**: `src/deckbuild.py` — 外部メタを取り込む代わりに**プールから直接コヒーレントなデッキを構築**（mono-type アグロ8種：強い基本ポケexの技持ち×各色＋一致基本エネ＋サンプル流の draw/search エンジン）。全て engine 合法。`scripts/build_demo_decks.py` で `decklists/*.csv` に出力（`tests/test_deckbuild.py`）。
- **デッキリスト取込（名前→ID）**: `src/decklists.py` — `<枚数> <カード名>` 形式の人間可読リストを ID 化（同名複数刷りは最小ID代表）。将来の実メタ取込の口（`tests/test_decklists.py`）。
- **本デッキ評価（`run_deck_eval.py --deck-dir`, Docker, greedy, 80戦/対, 10デッキ）**: デモ8種＋サンプル＋ランダム合法1。全コヒーレントデッキが**ランダムを ~1.00 で圧倒**（=実プレイ可能を確認。例外: psychic_aggro が ~0.45 と弱い外れ値）。**弱点に沿った相性構造**が出現（例: water>fire 0.80, metal>water 0.95）。周辺勝率: metal 0.744 / sample 0.717 / fighting 0.690 … psychic 0.103。**デッキ spread 0.683**（実デッキ同士でも大）＝「デッキ差≫エージェント差(~0.50 ミラー)」を**実デッキで再確認**。結果 `results/deckeval_demo_greedy.json`。
- **初回ラダー提出（準備完了・提出はユーザ操作）**: `submission/main.py`（**自己完結 greedy**：deck.csv 読込＋`all_attack()` から打点表を起動時生成＋全例外で合法フォールバック）。`scripts/build_submission.py` が `build/submission/`(main.py+deck.csv+cg/) を生成。Docker スモークで**自己対戦が98手で完走・クラッシュ0**を確認。既定デッキ=metal_aggro（ローカル最良）。**提出済み（2026-06-20, ref 53874111）**：検証 COMPLETE → 初期レート **≈602 / 1378位(2174)**（516.6→602.6で変動中）＝M1のローカル↔ラダー較正起点。提出手順・ハマり所は CLAUDE.md「Submitting to the ladder」。
- **未確認（follow-up）**: errorType=1 の意味（不明カードID? 未テスト）／「同名・別ID」を 5枚にしたケースの厳密確認（現プローブは同ID5枚で代理確認）。Radiant 等の特殊上限。psychic_aggro が弱い原因（攻撃役の選定 or 相性）。
- **次（Phase 3 へ）**: 構築物（プール/合法マスク/デモデッキ/評価ハーネス）を土台に、状態/行動表現＋ネット骨格（CB+BTヘッド）。実メタ取込は `src/decklists.py` 経由でいつでも追加可。

### Phase 2 メモ（ヒューリスティック評価関数の結論）

- **作ったもの**: `src/agents/heuristic_agent.py`。(1) **状態評価関数** `evaluate_state`（prize差・盤面HP・付与エネ・ベンチ展開・両者の即KO脅威を1スカラーに。探索のリーフ評価器/学習の価値教師を兼ねる）と、(2) それと同じ部品を使う**1手先読みなしの方策**。特徴は6フラグ（`attach_target`/`attack_ko`/`weakness`/`bench_dev`/`retreat`/`promote`）で**個別にon/off可能**＝ablation前提設計。エンジン由来データ（カード/技stats）はrunnerが注入し、エージェントは純粋な`dict->list[int]`のまま（`cg`非依存）。
- **結論（達成基準に対して）**: **未達**。`heuristic` は サンプルデッキのミラーで greedy と**互角**（0.50–0.53、CIが50%跨ぎ、N=1000でも0.496）。≥55%の明確採用バーに届かず。一方 random には 0.916 で圧勝＝方策自体は健全。1手~0.008ms/最悪0.06ms・クラッシュ0で提出衛生もOK。
- **特徴量ごとの寄与（ablation, dir=方向性のみ・全て有意未満）**: `promote`(+3.6pp) と `attach_target`(+3.2pp) が有効方向、`retreat`(−2.2pp) が有害方向（アグロでは退却の**エネ捨て**がテンポ損）、`attack_ko`/`weakness`/`bench_dev` は**中立**（方策の選択を変えない。弱点は一様倍率で攻撃 argmax 不変、提示される攻撃は既に合法＝最大打点を選ぶだけ）。
- **なぜ互角か（診断）**: サンプルデッキは40枚エネルギーの高速アグロで**平均~6.7ターン**で決着。両者とも develop→attack の核は同一、勝敗を分けるのは(a)引きの運と(b)多数の CARD サブ選択だが、後者の改善余地は薄い（probe結果: 大半が setup-active=promoteで対応済 / 複select benchは全展開 / prize選択は裏向きで不可視）。**ミラー対戦のため構造的にほぼ対称** → 盤面ヒューリスティックの上積み余地が小さい。
- **判断**: ここで僅差を追ってシード/構成を選ぶのは §A 違反（ノイズへの過剰適合）。**全特徴を「保留」**とし、デッキ確定後（Phase 1）に再計測する。これは「デッキ選択が支配的レバー」という本プランの中心仮説と整合（実デッキ＋多相手・非ミラーで評価関数の価値が出るはず）。
- **次の一手の候補**: ① **Phase 1（デッキ選定）を先に**回し、選定デッキ＋非ミラー相手で Phase 2 ablation を再走（最有力）。② サブ選択（サーチ/トラッシュでのカード価値付け）はこのデッキでは薄いが実デッキ次第で再検討。③ 構築物（評価関数・特徴・ablationハーネス）はそのまま **学習の価値教師(Phase4) / 推論時探索のリーフ評価(Phase6)** として流用可。

### Phase 3 メモ（状態/行動表現＆ネット骨格 — 完了）

- **作ったもの（`src/net/`）**: **方策+価値+CB ネット骨格**。`features.py`（engine 注入の **固定カード特徴**＝card-type/フラグ/HP/エネ型/弱点/最大打点を 40次元へ）/`encode.py`（観測→**固定長 state 191次元**、各 option→**63次元**。me/opp 視点を `yourIndex` で整列、欠損は全ゼロ＝**非クラッシュ**）/`model.py`（`PolicyValueNet`：**numpy** 共有トランク＋value/policy/CB の3ヘッド、`NetConfig` で全幅可変、重み **npz** 保存＝**推論/提出パス**）/`cb.py`（`legal_next_ids` マスク下で 60枚を逐次生成）/`nn.py`（numpy forward 用の he_init・softmax のみ）。**学習側**: `torch_model.py`（同一アーキの torch 版＋numpy 重み橋渡し）/`lit.py`（`LitPolicyValue`＝BC＋価値回帰の Lightning モジュール、可変 option 長は padding＋mask）。`src/agents/net_agent.py`＝**純 `dict->list[int]`** の `NetAgent`（提示 option を採点→top を返す＝常に合法、全例外で合法フォールバック）。レジストリに `net` 登録。
- **設計判断（学習=torch/Lightning, 推論=numpy の分離）**: **学習は torch + Lightning**（autograd・optimizer・checkpoint・将来の GPU/分散＝Phase5 OSFP に必須。`torch`/`lightning` は **dev 依存**に宣言）。**提出/推論は純 numpy forward を維持**（PLAN §D「推論依存最小・重みを numpy で読む・CPU・手番時間」に合致、バンドル軽量。Kaggle サンドボックスに torch がある保証もない）。橋渡し＝`torch_model.py` が**同一アーキを torch で持ち、重みを numpy dict に正確変換**（`to_numpy_net`/`from_numpy_net`）。**parity テストで torch↔numpy forward の数値一致(<1e-9)を保証**＝「torch で学習→npz エクスポート→numpy で serving」が安全に往復。論文の「学習カード埋め込み」は当面**固定特徴の学習射影（MLP）**として実現＝勾配が素直・未知IDは零ベクトルで頑健（学習埋め込み化は後段で ablation 可）。
- **行動分解の往復**: 論文の autoregressive `(type,target)` を、本コンペの**動的 option リストを直接採点**する形に落とした。engine は合法 option しか出さない＝**0/1合法マスクが自明に成立**し、option-index 契約に**ゼロコストで往復**。multi-select は score 上位 `maxCount` を返す（`legal_fallback` の `range(maxCount)` と同じ常時合法な形）。
- **実機検証（Docker, `scripts/probe_net.py`）**: ① **CBデッキ** greedy+sampled×2 を `battle_start` が**全て errorType=0 受理**（distinct names 15/59/56＝混合で多様）。② **net vs greedy 4局＋net vs net 2局＋run_eval 30局**で **crashes=0 / illegal=0**（~580選択）。③ **推論時間** avg **0.12ms**・最悪 **6.4ms**（greedy ~0.002ms の~50倍だが sub-ms 中心、手番制限に桁違いの余裕＝探索/価値リーフを載せる予算十分）。net params **21,987**。
- **学習配線サニティ（torch/Lightning, `tests/test_net_torch.py`）**: 合成データ（target=option特徴の argmax）で `LitPolicyValue.policy_loss` が **loss を初期比<0.7 へ低下・精度 chance 0.25→>0.5**。`Trainer.fit`→`to_numpy_net().save()`→`PolicyValueNet.load()` が往復し、**torch↔numpy forward の parity<1e-9**。padding マスクで余剰 option を確実に除外。テスト **99件 green**（ruff ALL / ty / pytest）。
- **強さは未（想定どおり）**: random 初期化なので net は greedy に **0/30**。Phase 3 は“配線”で強さは問わない（exit基準は合法 forward・合法CB・時間予算・学習配線の4点で**全達成**）。
- **次（Phase 4 へ / Phase 2 再走）**: ① **Phase 4 BC 暖機**＝`heuristic`/`greedy` の対戦ログ（観測,手,結果）を `(states,options,mask,targets,values)` バッチに整形し `LitPolicyValue` で BT/価値を学習（CBヘッドも同型の masked-CE で）。保険提出候補。② 並行して **Phase 2 ablation を実デッキ・非ミラーで再走**（heuristic 保留の解消）。骨格はそのまま OSFP(Phase5) の初期化に載る。

### Phase 4 メモ（行動クローン暖機 — 完了, CB決定的生成のみ制約）

- **作ったもの**: ①`scripts/collect_bc.py`（Docker・教師ログ収集）＋`run_eval.play_game` に後方互換な `recorder` フック（適用手の直後に obs を deep-copy 記録、終局で winner）。②`src/net/bc_data.py`（生ログ→`encode_state/encode_options` で 5-tuple、価値ラベル＝手番視点の勝敗±割引、`collate_*`、CB教師データ `cb_supervision`）。③`src/net/lit.py` に `cb_loss`＋`LitCB`（同一ネットの **cb1/cb2 のみ**最適化＝BT/価値と独立）。④`scripts/train_bc.py`（policy/価値→CB を1ネットで学習→`.npz` エクスポート、held-out 評価）。⑤`run_eval` に `--a/b-weights`・`--cb` 配線、`probe_net` に `--weights`＋CB重複レポート。⑥**循環import修正**: `encode.py` が `src.agents.base` から `AREA_*` を import していた箇所を局所ミラーに（net層を agents 非依存化＝`encode` が最初に import されても壊れない）。
- **学習データ**: heuristic 教師の **400局**（8デモデッキ総当たり×対戦相手 {heuristic,greedy}、`data/bc/`、156MB、収集 **11秒**/Docker）。単一選択判断のみ採用＝**23,261 policy samples**。学習はネイティブ torch+Lightning で **~11秒**（22k params, CPU）。両者の分離は「学習=torch/推論=numpy」方針（[[training-stack-torch-lightning]]）どおり、parityで往復保証済み。
- **結果（実機・metal_aggro 固定・各500局）**: net(heuristic-init) vs **random 0.998** / **greedy 0.532（同等）** / **heuristic 0.604（教師超え）**。held-out **policy top-1 0.874・value符号 0.895**。**crash 0 / illegal 0 / 最悪 0.87ms**。→ Exit の「random圧勝・教師同等以上・value偶然超・退行なし」を**全達成**。教師をわずかに超えるのは BC 平均化（設計の論理 pt4）。最終的な上積みは Phase5 RL の仕事。
- **教師 ablation（同一ログ流用）**: greedy 教師でも学習可（`teachers={greedy}`、7,004 samples、val acc 0.954）。実機 vs random/greedy/heuristic = 0.990/0.488/0.580。**全対戦で heuristic-init ≥ greedy-init** → 既定教師 = **heuristic** を確定（プラン既定と一致）。
- **価値ターゲット ablation（最終 vs 割引）について**: `--discount` つまみは実装済みだが、**NetAgent の対戦は policy ヘッドのみ使用**（価値ヘッドは推論で不使用）ため net 同士の勝率に影響しない＝この ablation は **Phase5（価値を advantage に使う）で評価**するのが正しい。Phase4 では value符号正解率で健全性のみ確認。
- **CB 所見（重要・アーキ起因の制約）**: CBヘッドは **固定特徴**で各カードを採点（`features.py`：学習カード埋め込みは後段ablation）。ゆえに「カード個体」でなく「**特徴プロファイル**」を順位付けし、**greedy decode は最高プロファイル＋cap免除の基本エネに崩壊**（distinct=2）。inverse-copy 重み（`CBSample.weight`＝1/枚数。エネ支配を抑える）でも個体識別不能の壁は残る。一方 **sampled decode は distinct=50 で合法・多様**（「デッキは分布」と整合）。→ **(a)** 決定的提出は **デモデッキを安全網**（§Phase7 安全網）、**(b)** OSFP(Phase5) では **sampled CB** を相手デッキ多様化に使える、**(c)** CB の個体選択力には **学習カード埋め込み**が必要（後段で ablation）。
- **提出衛生**: バンドルは numpy のみ（torch非依存）を維持。`bc_net.npz`=22k params/~180KB。`data/bc/`（ログ・engine.json・npz）は gitignore。
- **次（Phase 5 へ）**: `data/bc/bc_net.npz`（heuristic-init）を **OSFP 自己対戦の初期重み**に確定。Phase5 で「**BC暖機あり vs なし(from-scratch)**」を同計算量で実測（暖機の価値の裏取り＝ユーザ合意事項）し、価値ターゲット(最終 vs 割引)・デッキヘッド学習 on/off を ablation。

### Phase 5a メモ（OSFP 自己対戦 — 配線完了, 実RLラン待ち）

- **作ったもの**: ①`src/agents/net_agent.py` に**確率的サンプリング**（`temperature>0` で単一選択を softmax からサンプル、`reset(seed)` で per-game RNG。既定 `temperature=0`＝argmax で評価/提出挙動は不変。multi-select は決定的 top-k のまま）。②`src/net/lit.py` の **`LitPolicyGradient`**（REINFORCE＋価値ベースライン＋エントロピー。BCと**同じ5-tupleバッチ**を再解釈＝target は実際に取った手・value はリターン。`configure_optimizers` は **`cb*` を除外**＝CBヘッド凍結＝デッキ固定アーム）。③`src/net/osfp.py` の **`OpponentPool`**（純numpy・論文 Algorithm1 の縮約：直近重み付きサンプリング＋scriptedベースライン下限＋self-play確率、チェックポイント採用＝全相手を閾値超 or patience）。④`scripts/collect_selfplay.py`（Docker・`collect_bc.BCRecorder`/`play_game` 再利用、自己対戦は両スロット `"learner"`・対相手は学習者側のみタグ）。⑤`scripts/train_osfp.py`（**注入式の純ループ `run_osfp(cfg, feats, *, generate, evaluate)`**＝Docker無しで全ループをネイティブ検証可、本番は `docker run` を反復）。
- **設計の論理**: 1イテレーションで直近重みが生成した≒on-policyデータを1パス学習＝重要度比≈1なので**内側更新は REINFORCE で十分**（V-Trace/PPO は後段 ablation）。γ=1.0（`discount=None`）＝報酬はエピソード末の±1のみ。価値ヘッドは**RLベースライン専用**で学習し**推論では未使用**（`NetAgent.act` は policy のみ＝提出パス無変更）。
- **load-bearing なバグ修正（実装中に実測で判明）**: (a) **masked entropy の NaN** — `logp.exp()*logp` が masked 位置で `0*-inf=nan`、後から masked_fill しても**逆伝播で再発**（mul の grad が保存済み -inf を掛ける）。→ 積の**前に** `logp` を0埋め（`safe_logp`）。`entropy_coef=0` でも `0*nan=nan` で全損失が汚染されるため必須。(b) **中断局の混入** — `_outcome` は draw も abort も 0。`run_osfp` が `winner in (0,1)` でフィルタしてから `build_policy_samples`。
- **検証（ネイティブ）**: `ruff/ty/pytest` 全緑（**125 passed**）。`test_osfp.py`（プール論理11ケース）＋`test_rl.py`（PG: +advで logp↑/value回帰/エントロピーNaN安全/CB凍結、学習者タグ抽出、確率的agent合法・再現、export→NetAgent合法、`run_osfp` 全ループ）。
- **検証（Docker `--smoke` 実機・配線確認済み）**: `collect_selfplay` 8局~1s・全decisive・タグ正常（learner/opp 片側ずつ＝リーク無）。`train_osfp --smoke`（3iter×8局, 6.3s）が pool 選択→収集→PG学習→採用→export を完走（**自己対戦は両スロットlearnerで~2倍サンプル**＝タグ規約が機能、iter3 で patience 採用）。学習後 `final.npz` を実機 probe＝**crash0/illegal0/worst1.0ms・PASS**、**CBヘッド凍結を実証**（greedy distinct=2/sampled 50＝Phase4と一致）。
- **未計測（ユーザ起動の本RLラン待ち）**: **OSFP net vs BC/heuristic の 500局勝率（≥55%が合格）**、BC暖機あり/なし（from-scratch）の同計算量比較、recency/self_play_prob/entropy_coef/τ の ablation。本ランは `uv run python scripts/train_osfp.py --iterations N --games M`（多時間・Docker）。
- **次（5b）**: CBヘッドRL（各局でデッキ sample→勝敗逆伝播・相手デッキ多様化）＝デッキONアーム。CB の個体識別には学習カード埋め込みが要る（Phase4 所見）ため、5b と併せて検討。

### Phase 5d メモ（統合 joint OSFP — 実装・速度整備完了, 本RLラン稼働中: デッキ崩壊を特定・修正済）

- **方針の収束**: 5a(プレイ天井)・5b(文脈自由CB破綻)・5c(LSTM頭でも固定デッキ超え不可)を経て、**論文どおり「デッキ↔プレイを自己対戦で同時学習」**に一本化。**固定デッキ前提・vs固定スコアリングを撤去**（ユーザ指摘: 自己対戦なら試合は無限生成可＝「データ不足」は誤り。真因は(a)スコアリングをデッキ vs デッキ自己対戦に直す＋(b)計算速度）。
- **作ったもの**: `scripts/train_joint_osfp.py`（`run_joint_osfp`：1更新で **play+価値+デッキ+共有埋め込み**を凍結なし最適化）＋`scripts/collect_joint_selfplay.py`（1ゲームから play 遷移とデッキ W/L の両方を出力）。旧 `train_osfp`/`train_deck_osfp`/`collect_selfplay`/`collect_deck_selfplay` を**削除**。`LitJointPolicyGradient`＝`CombinedLoader{play,deck}`で両アーム同時。
- **共有埋め込み（論文の要）**: 学習カード埋め込みを**playヘッドにも注入**（`encode.py` が盤面ポケモン/各optionの対象カードの埋め込み行を出し、numpy/torch forward が全カード位置で lookup）。**parity<1e-9 維持**、`load()` は重み形状から**全層幅を復元**（任意サイズ可）。両ヘッドの損失が同一 `cb_embed` に勾配を流す＝真の共有表現。
- **デッキ品質の是正（5b/5cの崩壊を根本解決）**: ①**CB BC の type-target 重み**（タイプ別総重み=物理比率・タイプ内は等重み）→ sampled デッキが demo構成(31/12/17)に一致（`1/copies`はエネを11%に潰していた）。②**構成制約付き greedy decode**（各タイプをネット自身のsampled平均でcap）→ 決定的デッキが **energy=29/pokemon=15** と機能的（無制約 argmax は単エネを46枚に増幅して崩壊）。sampled は無制約のまま＝RLの探索自由を維持。**色不整合**（虹ポケ+単色エネ）は**ルールベースのデッキ探索制約**で対処予定（下記「ルール制約メモ」。RL自己対戦のデッキ信号は弱く"淘汰"には頼れないと判明）。
- **エンジン所見**: 実機検証で**現実型デッキ(低エネ+ドロー/サーチ)を完全サポート**（ドロー/サーチ効果が発火）。∴デッキ空間は本物のモダンポケカ＝低エネ archetype も成立しうる（ただしパイロット(エージェント)の力量とセット）。
- **play質は手書きでなくRL**: TCGベストプラクティス(掘ってから貼る/need-awareサブ選択)を heuristic に入れても**ベースライン超えず(撤回)**。1-ply heuristic は生産的に掘れない＝**play の上積みは πBT の RL が筋**（joint の価値の裏付け）。
- **速度整備（本RLランの前提）**: **GPU修復**（torch を cu124 ビルドに固定＝A100稼働, Linux x86のみ・Mac非影響）＋学習GPU化＋**play決定のサブサンプリング**（~2万/iter, encode前にコレクタで間引き）＋**obsコピー除去**（deepcopyを~9x減）で **1 joint iter 726→50s（14.5x）**。更新の GPU speedup 31x(小)/57x(0.44M大)。0.44Mネット訓練 GPU **60s**。**500iter≈一晩 / 1000iter≈14h**（元8.4日）。ボトルネックはエンジンでなく**毎iterのサンプル処理+学習**だった（GPUが効く所）。
- **本RLランの所見（run1崩壊→run2修正, 台帳 P5d-RL診断/修正）**: run1 は **デッキが iter2 で崩壊**（→台帳）。原因を多面調査で特定し、**デッキアームに entropy+BC-KLアンカー＋76×除去**で**崩壊解消**（run2: E∈[28,32]維持）。run2 の **gate は BC 0.175→iter25 0.55** と大改善するが、**改善は play頭(πBT)のみ**でデッキ構成は BC のまま（KL=1.0 が強く凍結＝grounding欠如で自己対戦のデッキ信号が弱い）。**デッキ品質を上げるのは"RLに任せる"でなく"ルール制約で注入"が筋**（下記）。
- **次**: (1) run2 を回し切り play頭の伸びを gate で追跡。(2) **デッキ品質はルール制約で注入**（下記メモ）。(3) デッキも RL で動かしたいなら deck_kl を下げて崩壊しない範囲を探す。(4) 伸びれば improved techniques(PPO/V-Trace)・更なる容量/台数。

### ルール制約メモ（ルールベースのデッキ探索制約 — やること）

**動機**: 自己対戦のデッキ学習信号は弱い（grounding欠如＝両者が一緒に崩壊→ミラー~0.5→advantage薄。run1崩壊・run2デッキ凍結で実証）。正則化(entropy+BC-KL)は**崩壊を止める**が、デッキ品質を**積極的に上げる**力はない。→ **デッキ探索に堅い TCG ルールを制約として課す**のが、品質を確実に注入する最短手（RLに色整合を"発見"させるより堅い）。具体的弱点: 初期BCデッキ自体が**色不整合**（単色エネ29＋4型に散ったポケモン15、型一致は4体のみ・残りは無色頼み）。

**仕組み（既にある土台）**: デッキ生成は**逐次マスク付きカテゴリカル選択**。各ステップで `legal_next_ids(部分デッキ, pool)` がマスクを作り、**decode（`cb.py:_decode_deck`）と deck損失（`bc_data.py:cb_sequences`）の両方が同じ関数を使う**。greedy の `caps`（型枚数上限）は既にある**ルール制約の実例**。
→ やることの核: **`allowed_next_ids(deck, pool, rules)` を1つ作り、decode と損失の両方をそれに差し替える**（**両者で同一マスク必須**＝REINFORCE の対数確率が同じ候補集合で正規化される）。

**入れるルール（1つずつ・§A ablation）**:
1. **色整合（最優先）**: 部分デッキの主エネルギー色にコミット→そのアタッカーの技コストを「主色＋無色」で払えないポケモン／別色の基本エネをマスク。今の虹デッキを直接潰す。
2. **進化ライン**: たね不在の1進化/2進化をマスク（自己回帰順で基本を先に許す）。
3. **エネルギー帯**: 上限（例35）到達でエネルギーをマスク／残枠で下限を確保（`_type_caps` の一般化）。
4. （将来）必須サポート／ドローエンジン最低枚数 等。

**正しさ・安全**:
- **decode と損失で同一マスク**（でないと REINFORCE が壊れる）。
- **各ステップで必ず ≥1枚 合法を残す**（制約で全消ししない／空なら合法へ soft フォールバック＝既存 caps と同じ運用）。
- **アーキ変更／BC再訓練 不要**: マスク差し替えで**現ネットでも即コヒーレントなデッキを生成**（off-color を 0 化するだけ）、その上で joint ループが適応。安く即効。

**補完案（C）**: BC/KL アンカー先を「8デッキ混成」でなく**単一コヒーレントデッキ（例 metal_aggro）**にすれば、初期デッキが最初から整合。ルール制約と併用可。

**注意（§A/§C）**: **締めすぎない**。ローカル sim はラダーを誤予測するので、ルールは**色整合・進化ライン等の TCG として堅い規則**に限定し、メタの好み（特定デッキが強い）は埋め込まない。**各ルールは ablation**（入れて gate/ラダーが上がるか）で採否を判定する。

---

## マイルストーン（目安）
- **M1**: P0–P1 完了 → 評価基盤＋**デッキ空間の足場（合法マスク/プール/メタデッキ・デモ）**＋初ラダー提出（ローカル↔ラダー較正開始）
- **M2**: P2–P4 完了 → ヒューリスティック土台＋表現/骨格（CB+BTヘッド）＋**BC暖機（プレイ＋デッキ）**（非ランダム学習方策の保険提出が可能に） ✅(BT/価値達成・教師超え。CB決定的生成のみ制約＝学習埋め込み待ち)
- **M3**: P5 完了 → **OSFP自己対戦ネット（デッキ↔プレイ同時学習）がラダーで前段を有意に上回る**（論文中核の達成）
- **M4**: P6–P7 完了 → 推論時探索の採否を確定＋頑健化した最終提出（〆切 2026-08-16）
