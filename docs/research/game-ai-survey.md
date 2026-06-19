# ゲームをプレイするAIエージェント — アルゴリズム/アーキテクチャ最新サーベイ（2020–2026）

> Pokémon TCG AI Battle（不完全情報・ターン制・CPU制約・先読み探索API有）に取り組むための調査。
> ディープリサーチ（24ソース取得→112主張抽出→上位25主張を3票の敵対的検証、23確認/2棄却）の結果を中核に、
> 検証バッチ外だが一次ソースが取得済みの領域を補って整理した。各主張の確度を `[確度: 高/中]` で明示する。

## 0. 全体像 — 3本柱

```
                 完全情報            不完全情報
              ┌────────────────┬──────────────────────────┐
  探索中心    │ MCTS/UCT       │ 決定化(PIMC) / ISMCTS     │ ← Pokémon TCGの探索APIはここ
              │ AlphaZero(MCTS)│ AlphaZe**(PIMC+NN)        │
              ├────────────────┼──────────────────────────┤
  学習中心    │ MuZero系       │ DeepNash(R-NaD, 探索なし)  │
 (self-play)  │ EfficientZero  │ ReBeL(RL+探索→ナッシュ)    │
              ├────────────────┼──────────────────────────┤
  ゲーム理論  │  —             │ CFR / Deep CFR / Pluribus │
              └────────────────┴──────────────────────────┘
  補足: Transformer方策(Decision Transformer) / LLMエージェント / ニューラル状態表現
```

- **探索系**は「いま手元にある計算（CPU・10分）で強くする」路線。学習済みモデル無しでも動く。**コンペ初手に最適**。
- **モデルベースRL（MuZero系）**は「オフラインで自己対戦学習し、推論は軽い方策＋短い探索」路線。GPU学習が前提だが推論は軽くできる。
- **不完全情報ゲーム理論（CFR/ReBeL/DeepNash）**は「隠れ情報下でナッシュ均衡（=付け込まれない戦略）に近づける」路線。理論的に最も筋が良いが実装重い。

---

## 1. 探索系 — MCTSと不完全情報拡張

### 1.1 基礎: MCTS / UCT
モンテカルロ木探索は「選択(UCBで有望ノードを辿る)→展開→プレイアウト→価値逆伝播」を予算いっぱい繰り返し、最も訪問された手を選ぶ。完全情報ゲーム（囲碁等）で実績。**ニューラル価値/方策を組み込むとAlphaZero**になる。

### 1.2 決定化 (Determinization / PIMC) と「戦略融合」問題
不完全情報を扱う素朴な方法が**決定化（PIMC: Perfect Information Monte Carlo）**。隠れ情報を確率的にサンプリングして「完全情報のゲーム」を複数生成し、各々を普通に探索して結果を平均する。

- **弱点1: 戦略融合 (strategy fusion)** — 同じ情報集合（自分には区別できない複数状態）の各状態が別々の木ノードになるため、「実際には区別できないはずの状態ごとに違う手を選べる」と**誤って仮定**してしまう。`[確度: 高 / Cowling+ 2012 一次論文を逐語確認]`
- **弱点2: 計算の無駄** — 決定化は独立した複数の木を作り情報を共有しないため、共通部分の探索が重複し予算が分散する。`[確度: 高 / 同上]`

### 1.3 ISMCTS (Information Set MCTS)
情報集合（=区別できない状態の集合）そのものを木のノードにして**単一の木**を探索する。手の統計が1つの木にプールされるため**計算予算を効率よく使い**、戦略融合の影響を「軽減〜完全に除去」できる。`[確度: 高 / Cowling, Powley, Whitehouse, IEEE TCIAIG 2012]`

**重要な注意 — 優位性はドメイン依存**: ISMCTSが決定化UCTに明確に勝つのは「深い探索が要る」「戦略融合が深刻」なゲーム（Lord of the Rings: Confrontation, Phantom 系）。一方、情報集合木の分岐因子が桁違いに大きいゲーム（Dou Di Zhu）では**決定化UCTと互角**にとどまる。`[確度: 高 / 同上、著者自身の結論]`

> ⚠️ 「ISMCTS×Pokémon」の研究（Ihara+ 2018, IEEE SMC）は存在し、決定化MCTSを頭対頭 57.5% vs 42.5% で上回ったと報告。ただしこれは**ビデオゲームのバトル（対戦システム）であって、本コンペのカードゲーム(TCG)とは別物**。知見は概念的示唆にとどめるべき。`[確度: 高（事実）/ ただし対象ドメインが異なる点に注意]`

### 1.4 NN × 探索のハイブリッド（最新）
- **AlphaZe** (Blüml, Czech, Kersting, Frontiers in AI 2023)** — AlphaZeroの**MCTSをPIMC変種に置換しただけ**のモデルベース型ベースライン。「AlphaZero的手法は不完全情報に弱い」という通説に反し**驚くほど強い**と報告（Barrage Stratego/DarkHex/Hex/Chessで検証）。ただしSOTAではなく「強いベースライン」（P2SROには負け、DeepNashには及ばない）。`[確度: 高 / 一次論文]` → **探索API＋ニューラル価値という本コンペの構成に最も直接的な前例。**
- **情報集合サンプリングで価値関数を学習 (Bertram, Fürnkranz, Müller, KI 2024 / arXiv:2407.05876)** — 「不完全状態の価値＝情報集合内の整合する全状態の価値の組合せ」と定義し、それを直接予測するNNを学習。固定予算なら「**少数の状態(≈2–3)を多数の異なる局面でサンプル**する方が、少数局面で多数サンプルするより良い（多様性＞ターゲット品質）」。`[確度: 中 / 2-1の分裂投票。これは“プレイ時の探索予算”でなく“オフライン訓練ターゲット生成”の話。検証2領域のみ]`

### 1.5 本コンペ(Pokémon TCG)への当てはめ
- 提供される `search_begin/search_step` は決定化（隠れ情報を予測入力）して木探索する設計 → **PIMC/ISMCTS系がそのまま乗る**。CPUでも動く現実解。
- まず**決定化＋短い探索＋ヒューリスティック評価**でベースライン、次に**ISMCTSで戦略融合を緩和**、さらに**ニューラル価値**を足す（AlphaZe**型）が王道の段階的拡張。
- 分岐因子（TCGは1ターンの選択肢が多い）が大きいと決定化UCTと差が出にくい点は要計測。

---

## 2. 強化学習・自己対戦

### 2.1 AlphaZero（2018, Science）
ルールのみを与え、**自己対戦だけのtabula rasa学習**。単一アルゴリズムでチェス・将棋・囲碁を24時間以内に超人レベル、各世界チャンピオンプログラムに勝利。`[確度: 高 / Silver+ Science 2018]`（※「tabula rasa/24時間/Stockfish撃破」には旧版Stockfish・非対称ハード等の学術的批判もあるが、論文記述自体は正確）
- アーキ: 残差CNN（盤面）→ 方策ヘッド＋価値ヘッド。MCTSの探索結果を教師に方策/価値を回す。

### 2.2 MuZero（2020, Nature）
**ルールを与えずモデル自体を学習**。学習モデルは環境全体を再構成せず「**計画に必要な3量＝報酬・方策・価値**」だけを潜在空間で反復予測し、その潜在空間上でMCTSを回す。`[確度: 高 / Schrittwieser+ Nature 2020]`
- 3ネットワーク: **表現(observation→潜在状態)・ダイナミクス(潜在状態+行動→次潜在状態+報酬)・予測(潜在状態→方策+価値)**。
- ルール不供給でもAlphaZeroの超人性能に匹敵（囲碁/チェス/将棋）＋Atariも制覇。`[確度: 高]`

### 2.3 Stochastic MuZero（2022, ICLR）
MuZeroの決定論モデルを**確率的環境**へ拡張。**afterstate**を導入し確率的木探索を行うことで、本質的に確率的・部分観測な環境に対応。2048・バックギャモンでSOTA匹敵/超え、囲碁の超人性能は維持。`[確度: 高 / Antonoglou+ ICLR 2022]`
→ **Pokémon TCGのコイントス・山札シャッフル等の確率性に概念的に適合。**

### 2.4 EfficientZero V2（2024, ICML Spotlight）
MuZero/EfficientZero系譜で**限られたデータでの高サンプル効率**を狙う。離散・連続制御の両方で、従来SOTAのモデルベース手法**DreamerV3を66タスク中50タスクで上回る**。古典MCTSを**サンプリングベースのGumbel探索**に置換。`[確度: 高 / Wang+ ICML 2024]`
→ サンプル効率は魅力だが、学習・探索の計算コストはCPU制約下では要検討。

### 2.5 実装の出発点: LightZero（NeurIPS 2023 D&B Spotlight）
**AlphaZero/MuZeroファミリーを単一PyTorchで統合**（AlphaZero, MuZero, Sampled/Stochastic/Gumbel MuZero, EfficientZero, ReZero, UniZero…）。MCTSコアはPython/C++両対応。`[確度: 高 / 公式リポジトリ＋NeurIPS proceedings]`
→ **自前でMuZero系を試すなら実装の起点として有力。**

### 2.6 補足: モデルフリー自己対戦／リーグ学習／方策蒸留
`[以下は検証バッチ外。確立した事実だが本サーベイの敵対的検証は未通過。確度: 中]`
- **PPO自己対戦** — 実装容易で安定。多くのKaggle Simulations上位が採用（後述）。
- **AlphaStar (StarCraft II) / OpenAI Five (Dota2)** — **リーグ学習 / population-based training**で多様な相手を生成し、特定戦術への過剰適合（じゃんけん的な相性負け）を防ぐ。本コンペのラダー（多様な相手）と問題意識が一致。
- **方策蒸留 (policy distillation)** — 重いモデル/探索の出力を軽量モデルに模倣学習で写す。**オフラインで強く学習→提出は軽量推論**という、CPU・時間制約への定石的対処。

### 2.7 本コンペへの当てはめ
- フル自己対戦RLは**GPU学習が前提**で、ローカルがCPUのみ＆simがLinux専用な本環境では学習基盤の用意がボトルネック。
- 現実的には「**探索（§1）を主役に、価値/方策だけをオフライン学習して軽く差す**」ハイブリッド、もしくは**方策蒸留で軽量化**が筋。Stochastic MuZeroの確率対応・afterstateは設計のヒント。

---

## 3. 不完全情報ゲーム理論 — 均衡近似

### 3.1 CFR / Deep CFR
`[CFR/Deep CFR/Pluribusは検証バッチ外。一次ソース取得済み。確度: 中]`
- **CFR (Counterfactual Regret Minimization)** — 各情報集合で「反実仮想後悔」を最小化する反復で、2人ゼロ和の**ナッシュ均衡に収束**。ポーカーAIの基盤。素のCFRはゲーム木を陽に走るため**大規模ゲームは抽象化(abstraction)が必須**。
- **Deep CFR (Brown, Lerer, Gross, Sandholm, ICML 2019)** — 抽象化なしに**NNで後悔を関数近似**し大規模ゲームへスケール。`出典: proceedings.mlr.press/v97/brown19b`
- **Libratus / Pluribus (Brown & Sandholm, Science 2017/2019)** — ヘッズアップ→多人数ノーリミットポーカーで超人。Pluribusは**比較的安価な計算**で6人ポーカーをプロ超え。`出典: science.org/aay2400`

### 3.2 ReBeL（2020, NeurIPS）
**自己対戦RL＋探索**の一般フレームワーク。任意の2人ゼロ和ゲームで**ナッシュ均衡に証明付き収束**し、AlphaZero型が苦手な不完全情報に適用可。ヘッズアップ・ノーリミットポーカーで**従来(Libratus/Pluribus)よりはるかに少ないドメイン知識で超人**（プロを165±69 mbb/gで撃破）。`[確度: 高 / Brown+ NeurIPS 2020]`
→ **「探索を要する」点が本コンペの探索APIと整合的。** ただし均衡計算のための信念状態(public belief state)の扱いは実装が重い。

### 3.3 DeepNash / R-NaD（2022, Science）
**プレイ時に探索を一切使わない**モデルフリー深層RLでStrategoをマスター。中核は **Regularised Nash Dynamics (R-NaD)**：均衡の周りを「循環」せず、マルチエージェント学習ダイナミクスを直接修正して**近似ナッシュに収束**（KL/エントロピー正則化でlast-iterate収束）。Strategoで既存SOTA AIを破り、対人プラットフォームGravonで**全期間トップ3**（対人勝率84%）。`[確度: 高 / Perolat+ Science 2022]`
→ **推論時に探索不要＝CPU・時間制約に極めて優しい。** ただし学習側は大規模。

### 3.4 本コンペへの当てはめ
- 隠れ情報（相手手札・山札・サイド）下で**付け込まれない戦略**を作る点で理論的本命だが、CFR/ReBeLは実装・計算が重い。
- **DeepNash型（推論時探索なし＝軽い）は本番制約に好相性**だが学習コストが高い。まずは§1探索系で土台を作り、相手モデリング/均衡的発想は段階的に。

---

## 4. 補足: Transformer方策・LLMエージェント・状態表現
`[検証バッチ外。一次ソース取得済み。確度: 中〜低]`
- **Decision Transformer 系** — 強化学習を「(報酬・状態・行動)系列の系列予測」として解く。所望リターンを条件にTransformerが行動を自己回帰生成。オフラインRL向き。`出典: arXiv:2205.15967 (Online Decision Transformer) 等`
- **LLMエージェント** — 推論・計画・ツール利用でゲームを解く試みが急増（survey: git-disl/awesome-LLM-game-agent-papers, arXiv:2309.17277 等）。ただし**1手ごとの推論コストが大**で、CPU・10分/試合・ネット遮断の本コンペでは**実戦投入は非現実的**（§前回の結論どおり、LLMはオフライン設計役に回す）。
- **ニューラル状態表現** — カードの埋め込み、場/手札/サイドの集合をエンコード（Deep Sets / 注意機構）。Lux AI等の上位は盤面を多チャンネル画像化しCNNで処理する設計が定番。

## 5. Kaggle Simulations 実戦解法の傾向
`[実務ブログ/フォーラム/解法リポジトリ由来。確度: 中]`
- **Lux AI 2021 上位（Pressman）** — Deep RL（IMPALA/PPO系の自己対戦）＋盤面のCNN表現。`出典: github.com/IsaiahPressman/Kaggle_Lux_AI_2021`
- **Hungry Geese 1位** — 自己対戦で鍛えた方策＋探索の併用。`出典: kaggle.com/competitions/hungry-geese/discussion/263655`
- **動向まとめ（nagiss）** — Kaggle Simulations系では「自己対戦RL」と「探索（手作り/学習価値）」の二大潮流＋両者ハイブリッド。`出典: speakerdeck.com/nagiss`
- 総じて: **計算資源と問題構造次第で「ルールベース＋探索」「自己対戦RL」「両者の蒸留ハイブリッド」のどれかに収束**。CPU提出・探索API有・不完全情報という本コンペは**探索系を軸にしたハイブリッドが相性良い**。

---

## 6. 手法比較表

| 手法 | 学習要否 | 推論時探索 | 不完全情報対応 | 推論コスト(CPU) | Pokémon TCG適性 |
|---|---|---|---|---|---|
| ルールベース | 不要 | 任意 | 手書き | 極小 | ◎ 初手・フォールバック |
| 決定化(PIMC) | 不要 | 必要 | サンプリング(戦略融合あり) | 中 | ○ 探索APIに直結 |
| ISMCTS | 不要 | 必要 | 情報集合木(融合緩和) | 中〜大 | ○〜◎ ドメイン依存 |
| AlphaZe**(PIMC+NN) | 要(価値/方策) | 必要 | PIMC+NN | 中 | ◎ 前例として有力 |
| AlphaZero | 要(自己対戦) | 必要(MCTS) | × 完全情報前提 | 中 | △ 完全情報向け |
| MuZero系 | 要(自己対戦,重) | 必要(MCTS) | Stochasticで部分対応 | 中 | △〜○ 学習基盤が要る |
| EfficientZero V2 | 要(高効率) | 必要(Gumbel) | 部分 | 中 | △ サンプル効率は魅力 |
| CFR/Deep CFR | 要 | 不要(均衡表/NN) | ◎ 均衡 | 小〜中 | △ 実装重い |
| ReBeL | 要 | 必要 | ◎ 均衡収束 | 大 | △〜○ 探索APIと整合だが重い |
| DeepNash(R-NaD) | 要(重) | **不要** | ◎ 均衡 | **小** | ○ 推論軽い/学習重い |
| Decision Transformer | 要(オフライン) | 不要 | データ依存 | 中 | △ |
| LLMエージェント | 事前学習済 | — | 推論で | **特大** | × 本番制約で非現実的 |

---

## 7. 最新トレンドと今後
1. **学習モデル＋探索の融合が主流**（MuZero, ReBeL, AlphaZe**）。「探索で短期、学習で長期/評価」を分業。
2. **確率性・部分観測への正面対応**（Stochastic MuZero のafterstate、不完全情報の信念状態）。
3. **サンプル効率と軽量化**（EfficientZero V2, 方策蒸留）— 限られた計算でSOTAに近づける流れ。
4. **推論時に探索を要さない均衡学習**（DeepNash/R-NaD）— 本番制約に強い方向。
5. **オープン実装の成熟**（LightZero）で、MuZero系の再現/転用の敷居が低下。
6. **LLM/Transformerはまだ「設計者・オフライン」役**が現実的（推論コスト）。

---

## 8. Pokémon TCGコンペへの実践的示唆（要約）
1. **初手**: 強いデッキ＋ルールベース`agent()`で合法手・ベースライン（クラッシュ厳禁）。
2. **第2段**: `search_begin/step`で**決定化＋短い木探索**＋ヒューリスティック評価関数。CPUで回る範囲で。
3. **第3段**: **ISMCTS化**で戦略融合を緩和（分岐因子が大きいと効果は限定的→要計測）。
4. **第4段**: オフラインで**ニューラル価値/方策を学習**し探索に差す（**AlphaZe**型**）。重ければ**方策蒸留で軽量化**。
5. **発想の引き出し**: 確率性は**Stochastic MuZeroのafterstate**、付け込まれにくさは**均衡(DeepNash/ReBeL)**の考え方を参照。
6. **評価**: ローカルsimは実ラダーを外す→**5サブ/日で実測しながら反復**。リーグ学習的に**多様な相手で過剰適合を避ける**。

---

## 9. 検証メモ（何が裏取り済みか）
- §1（ISMCTS/決定化/戦略融合/AlphaZe**/情報集合価値学習）、§2（AlphaZero/MuZero/Stochastic MuZero/EfficientZero V2/LightZero）、§3.2 ReBeL・§3.3 DeepNash は**一次論文を逐語確認した高確度**（一部 中確度を明記）。
- §3.1（CFR/Deep CFR/Pluribus）、§4（Transformer/LLM）、§5（Kaggle解法）は**一次/実務ソースは取得済みだが本調査の3票敵対的検証は未通過**。事実関係は確立しているが、数値や最新性は各出典で再確認推奨。
- **棄却された主張**（除外済み）: 「ISMCTSがPokémon TCGへスケール実証」（対象は別物のビデオゲーム）、「AlphaZe**のPC-PIMCがメモリ効率で選ばれた/方策を合成する」（根拠不十分）。

## 出典（主要）
- ISMCTS: Cowling, Powley, Whitehouse, IEEE TCIAIG 2012 — https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf
- ISMCTS×Pokémon(ビデオゲーム): Ihara+ IEEE SMC 2018 — https://ieeexplore.ieee.org/document/8616371/
- AlphaZe**: Blüml, Czech, Kersting, Frontiers in AI 2023 — https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1014561/full
- 情報集合サンプリング価値学習: Bertram+ KI 2024 — https://arxiv.org/abs/2407.05876
- AlphaZero: Silver+ Science 2018 — https://www.science.org/doi/10.1126/science.aar6404
- MuZero: Schrittwieser+ Nature 2020 — https://www.nature.com/articles/s41586-020-03051-4
- Stochastic MuZero: Antonoglou+ ICLR 2022 — https://openreview.net/pdf?id=X6D9bAHhBQ1
- EfficientZero V2: Wang+ ICML 2024 — https://arxiv.org/abs/2403.00564
- LightZero: opendilab, NeurIPS 2023 D&B — https://github.com/opendilab/LightZero
- ReBeL: Brown+ NeurIPS 2020 — https://arxiv.org/abs/2007.13544
- DeepNash: Perolat+ Science 2022 — https://arxiv.org/abs/2206.15378
- Deep CFR: Brown+ ICML 2019 — https://proceedings.mlr.press/v97/brown19b/brown19b.pdf
- Pluribus: Brown & Sandholm, Science 2019 — https://www.science.org/doi/10.1126/science.aay2400
- Decision Transformer(Online): arXiv:2205.15967 — https://arxiv.org/pdf/2205.15967
- LLMゲームエージェントsurvey: https://github.com/git-disl/awesome-LLM-game-agent-papers
- Kaggle Simulations動向(nagiss): https://speakerdeck.com/nagiss/kagglesimiyuresiyonkonpenodong-xiang
- Lux AI 2021解法(Pressman): https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- Hungry Geese 1位: https://www.kaggle.com/competitions/hungry-geese/discussion/263655
