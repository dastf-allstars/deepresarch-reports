

# **動画へのキャプション付与に関する研究：2025年7月現在の研究課題とトレンド**

## **1\. はじめに**

動画コンテンツの爆発的な増加に伴い、その内容をテキストで記述する動画キャプション付与技術の重要性が増していることが観察される。これは単なる音声の文字起こしを超え、動画の内容理解、検索、要約、アクセシビリティの向上、さらには人間とAIのインタラクションにおいて不可欠な要素となっている 1。動画コンテンツの総量が増加し続ける中で、その情報を効率的に管理し、多様なユーザーに提供するための基盤技術として、キャプション付与は戦略的な意義を持つ。

Wistiaのデータによると、キャプションは同社ユーザーが動画に追加するアクセシビリティ機能として最も多く、2021年以降その使用が572%増加している 4。この顕著な増加は、動画キャプションが単なる技術的機能ではなく、ビジネスおよび社会的な要請に応える戦略的ツールとして認識されていることを示唆している 5。この現象の背景には、アクセシビリティの確保という社会的要請と、ビジネス上の価値創出という二つの側面が相互に作用していることが考えられる。キャプションの利用増加は、主に聴覚障害者支援によって推進されているが 5、同時に検索エンジン最適化（SEO）やグローバルリーチの拡大といったビジネス上の利点も提供している 5。検索エンジンはテキストをインデックス化するため、キャプションを追加することで動画の可視性とランキングが向上し、関連キーワード検索による発見可能性が高まる 5。また、YouTubeの視聴者の80%が米国以外であるという報告は、国際的な視聴者にとって多言語字幕が重要であることを強調している 5。企業が動画制作予算を増やしている傾向も 4、動画コンテンツの重要性の高まりと、それに伴うキャプションの価値向上を示唆している。したがって、アクセシビリティ要件（法的規制を含む）が技術開発と導入を加速させ、それが結果的にコンテンツの発見可能性向上や国際展開といったビジネスメリットをもたらすという正のフィードバックループが存在する。初期のアクセシビリティ投資が、広範なビジネス価値を生み出していると解釈できる。このため、キャプション技術の研究開発は、純粋な技術的課題だけでなく、社会的責任と市場機会の両方を考慮した多角的な視点から推進されるべきである。

本報告書は、2025年7月現在における動画キャプション付与に関する最先端の研究課題と技術トレンドを包括的に分析することを目的としている。主要な技術動向、克服すべき研究課題、および今後の方向性を提示し、この分野の研究者、開発者、および意思決定者にとっての戦略的な指針となることを目指す。

## **2\. 動画キャプション付与の主要技術動向（2025年7月現在）**

### **2.1. 大規模言語モデル（LLMs）および大規模マルチモーダルモデル（LMMs）の進化**

大規模マルチモーダルモデル（LMMs）は、特に短尺動画のキャプション付与において優れた性能を示しており、視覚的特徴とテキスト的特徴を連結してLLMに入力することで、動画内容の包括的かつ文脈豊かな記述を生成する 10。この技術は、動画コンテンツの自動理解と記述において大きな進歩をもたらしている。

しかし、長尺動画への対応には依然として課題が残されている。LMMsは長尺動画の入力処理能力は向上しているものの、300語を超える詳細なキャプションを安定して生成することには困難を抱えている 10。この出力長の制限は、トレーニングデータにおける長尺キャプションの不足が主な要因であると分析されている 10。動画の長さが増加するにつれて視覚情報の量も増加するが、LMMsのコンテキストウィンドウがすべての関連詳細を捉えるのに不十分であるため、出力が不完全または過度に簡潔になる傾向が見られる 10。この課題に対処するため、LongCaption-Agentのような自動化されたフレームワークが提案されている。これは、既存のLMMsとLLMsを活用し、フレームレベル、クリップレベル、動画レベルの3段階で情報を抽出し、要約することで長尺キャプションデータを合成するものである 10。このフレームワークに基づいて構築されたLongCaption-10Kデータセットは、10,000の長尺キャプション例を含み、LMMsが1,000語を超えるキャプションを生成する能力を解放することに成功している 10。

長尺動画における時間的推論（Temporal Reasoning）の進展も重要な動向である。TemporalVLMのような動画LLMは、長尺動画における効果的な時間的推論と詳細な理解を可能にしている 12。また、Seq2Timeは、密な動画キャプション付与や時間的グラウンディングを統一システムで実現するために、時間的認識が不可欠であると述べている 12。NVIDIA AI Blueprintは、長尺動画の理解のためにVLMとLLMを組み合わせ、動画を小さなチャンクに分割し、それぞれに密なキャプションを生成した後、LLMで要約・集約する戦略を採用している 1。この戦略はライブストリームにも適用可能であり、継続的に動画チャンクセグメントを生成し、VLMがキャプションを生成し、LLMがそれらを要約・集約する仕組みである 1。LMMsの性能向上は、単にモデル規模を拡大するだけでなく、データ生成戦略（合成データ）と、効率的な入力処理戦略（チャンク化、RAG）の組み合わせによって達成されつつある。これは、データ拡張とアーキテクチャの工夫が、モデルの基本能力を補完し、新たな応用範囲を切り開いていることを示している。特に、時間的推論の強化は、動画という動的なメディアの特性を深く理解するために不可欠な要素である。長尺動画対応は、監視、教育、エンターテイメントなど、多岐にわたる実世界アプリケーションへのLMMsの適用可能性を大きく広げる。また、LongCaption-Agentのような合成データ生成技術の進展は、高コストな手動アノテーションのボトルネックを緩和し、研究開発の速度を加速させる可能性がある。

Thought-Augmented Fine-Tuningのような応用も注目される。CaptionT5は、LLM（T5）を動画キャプション付与のためにファインチューニングするモデルであり、「Show, Think, and Tell」という人間のような思考プロセスを模倣する「思考拡張ファインチューニング（Thought-Augmented Fine-Tuning）」を導入している 13。これは、CLIPのような事前学習済みVLMを用いて動画の視覚的概念（オブジェクト、アクション）を「思考」として抽出し、それに基づいてキャプションを生成するものである 13。このアプローチは、少数の動画フレーム（2〜4フレーム）でも高い効果を発揮することが示されている 13。

### **2.2. マルチモーダル融合技術の深化**

動画キャプション付与は、複数のモダリティ（視覚、音声、テキスト）を考慮することで、より正確で包括的な記述を生成することを目指している 14。この分野では、多様な情報源を統合する能力が、キャプションの品質を決定する上で極めて重要であると認識されている。

従来の動画キャプション研究は主に視覚コンテンツに焦点を当ててきたが、TikTokや映画などのユーザー生成コンテンツ（UGC）では、音声信号（スピーチ、音楽、環境音、効果音）が視覚的に推測できない重要な情報を提供するため、音声と視覚の相互作用が意味理解の中心となる 16。このような背景から、音声・視覚統合によるオムニモーダル理解の重要性が高まっている。MV-GPTは、未ラベル動画から学習するための新しい事前学習フレームワークであり、視覚フレームとASR（自動音声認識）によって文字起こしされたスピーチを組み合わせてキャプションを予測する 15。このモデルは、将来の発話をキャプションターゲットとして利用する双方向生成目的を採用することで、未ラベル動画のキャプション不足を克服する 15。UGC-VideoCaptionerは、短尺UGC動画の詳細なオムニモーダルキャプション付与のために設計されたモデルであり、音声と視覚モダリティのバランスの取れた統合を重視している 16。このモデルは、Gemini-2.5-Flashから蒸留された3Bパラメータのキャプションモデルであり、音声のみ、視覚のみ、および音声・視覚結合のセマンティクスをカバーする3段階のヒューマン・イン・ザ・ループパイプラインを通じてアノテーションされたTikTok動画を利用している 16。動画の「全体的な意味理解」には、視覚だけでなく音声の統合が不可欠であるという認識が深まっている。これは、従来の視覚中心のアプローチでは捉えきれなかった、より豊かな文脈情報を取り込むための進化である。

拡散モデルの応用も新たなトレンドとして浮上している。拡散モデルは、画像、動画、音声合成などの連続信号生成タスクで顕著な成功を収めてきたが、キャプションの離散的な性質と複数モダリティにわたる条件付き生成の複雑さから、動画キャプション付与への適用はこれまであまり注目されていなかった 20。MM-Diff-Netは、拡散モデルをマルチモーダル動画キャプション付与に適用し、複数のミッドフュージョン技術と「生成された記述」という新しい入力モダリティを導入することで、キャプション品質の向上を示している 20。実験結果は、すべてのモダリティを組み合わせることで最高のキャプションが得られることを示唆している 20。拡散モデルの導入は、生成モデルの多様性と品質を向上させる新たな道を開き、特に複雑な条件付き生成においてその強みを発揮し始めている。オムニモーダルキャプションは、動画コンテンツのより深い理解を可能にし、より自然で人間らしい記述生成に貢献する。拡散モデルの応用は、キャプション生成の品質と多様性を飛躍的に向上させる可能性を秘めており、将来的にはより創造的なコンテンツ生成にも繋がる可能性がある。

### **2.3. ストリーミング・リアルタイムキャプション技術の発展**

密な動画キャプション付与（Dense Video Captioning, DVC）は、動画内のイベントを特定し、それぞれに記述的なキャプションを生成するタスクである 3。この分野では、特にリアルタイム性が求められるアプリケーションにおいて、技術的な進歩が顕著である。

CVPR 2024で発表された研究では、ストリーミング密動画キャプションモデルが提案されている 23。このモデルは、動画全体を処理する前に予測を生成するストリーミングデコーディングアルゴリズムと、任意の長さの動画を処理できる固定サイズのメモリモジュール（トークンクラスタリングに基づく）を特徴としている 23。これにより、ActivityNet、YouCook2、ViTTのベンチマークで最先端（SOTA）性能を大幅に向上させている 23。従来のDVCモデルは全動画処理後に単一予測を行うため、長尺動画やリアルタイム用途には不向きであった。ストリーミング機能の導入は、ライブイベントのキャプション、監視システム、リアルタイムの人間-ロボットインタラクション（HRI）など、即時性が求められるアプリケーションでの動画キャプションの適用範囲を大幅に拡大する。特に、固定サイズのメモリモジュールとストリーミングデコーディングアルゴリズムの組み合わせは、計算リソースの効率的な利用と低遅延なキャプション生成を可能にする。

### **2.4. 圧縮動画からの直接キャプション生成**

従来の動画キャプション付与アプローチは、まずデコードされた動画からフレームをサンプリングし、その後特徴抽出やキャプションモデル学習を行うというパイプラインを採用していた 25。しかし、この方法では手動でのフレームサンプリングがキー情報を見落とし、パフォーマンスを低下させる可能性があった 25。また、サンプリングされたフレーム内の冗長な情報が推論効率の低下を招いていた 25。

ICCV 2023で発表された研究では、この課題に対処するため、圧縮ドメインから直接動画キャプションを生成する新しいアプローチが提案されている 25。この手法は、Iフレーム、モーションベクトル、残差から構成される圧縮動画を活用することで、動画全体を学習に利用し、手動サンプリングの必要性をなくす 25。提案されたエンドツーエンドのTransformerモデルは、シンプルながらも様々なベンチマークで最先端の性能を達成し、既存のアプローチよりも約2倍高速に動作することが示されている 25。このアプローチは、動画処理パイプラインの効率化と高速化に貢献する。特に、大規模な動画データセットやリアルタイム処理が求められるシナリオにおいて、計算コストと処理時間の削減は大きな利点となる。圧縮ドメインからの直接学習は、データの前処理ステップを簡素化し、エンドツーエンドの学習をより効率的に実現する。

### **2.5. 事前学習済み画像-テキストモデルの動画キャプションへの応用**

動画キャプションモデルの開発は、計算コストが高く、動画の動的な性質がマルチモーダルモデルの設計を複雑にするという課題がある 26。このため、効率的なモデル開発手法の探索が重要視されている。

しかし、最小限の計算リソースと動画の動的特性に対処するための複雑な変更なしに、画像ベースのモデルを再利用して、いくつかの専門的な動画キャプションシステムを上回る性能を発揮できることが発見されている 26。具体的には、BLIP-2のような典型的な画像キャプションモデルを、わずか6,000組の動画-テキストペアで後学習し、単純にフレームを連結するだけで、競争力のある動画キャプションモデルに変換できる 26。これは、他の手法が250万から1億4400万ペアものデータを使用しているのと比較して、はるかに少ないデータ量である 26。この研究は、リソース最適化の観点から、モデル規模の最適化、データ効率の最大化、強化学習の組み込みという3つの基本的な要因に焦点を当てている 26。中規模の言語モデルが特定のタスクにファインチューニングされた場合に、大規模モデルに匹敵する、あるいはそれ以上の性能を発揮できるという結果は、常に大規模モデルが優れているという一般的な認識に異議を唱えている 26。このアプローチは、低リソース環境や、動画固有の複雑な設計がボトルネックとなる場合に実用的な解決策を提供する。画像-テキスト事前学習の知識を動画タスクに転移させることで、既存の強力なモデルの能力を効率的に活用し、開発コストと時間を削減しながら高性能なキャプション生成を実現する道が開かれる。

## **3\. 動画キャプション付与の主要研究課題（2025年7月現在）**

### **3.1. 長尺動画における詳細かつ事実に基づいたキャプション生成の困難さ**

大規模マルチモーダルモデル（LMMs）は短尺動画のキャプション付与で優れた性能を示すものの、長尺動画に対して300語を超える詳細なキャプションを安定して生成することは依然として大きな課題である 10。この出力長の制限は、トレーニングデータセットに長尺キャプションの例が不足していることが主な原因であると特定されている 10。手動で長尺キャプションをアノテーションする作業は、時間とコストがかかるため、このボトルネックが研究の進展を妨げている 10。

動画の長さが増加するにつれて視覚情報の量も増加するが、LMMsのコンテキストウィンドウがすべての関連詳細を捉えるのに不十分であるため、出力が不完全または過度に簡潔になる傾向が見られる 10。この課題に対処するため、LongCaption-Agentのような自動化されたフレームワークが提案されている。これは、既存のLMMsとLLMsを活用し、フレームレベル、クリップレベル、動画レベルの3段階で情報を抽出し、要約することで長尺キャプションデータを合成する 10。このフレームワークに基づいて構築されたLongCaption-10Kデータセットは、10,000の長尺キャプション例を含み、LMMsが1,000語を超えるキャプションを生成する能力を解放することに成功している 10。この問題は、動画キャプションが単なる短い説明ではなく、動画全体の物語や複雑なイベントシーケンスを詳細に記述する能力が求められるようになった現代において、特に顕著である。長尺動画の真の理解と生成には、単一フレームや短いクリップの情報を集約するだけでなく、時間的な依存関係や複数のイベント間の関係性を捉える高度な時間的推論能力が不可欠となる。

### **3.2. キャプションの「幻覚（Hallucination）」問題**

動画キャプション付与のようなテキスト生成タスクでは、入力動画によって裏付けられない事実誤り、すなわち「幻覚」を導入するリスクがある 28。これらの事実誤りは、生成されたテキストの品質に深刻な影響を与え、時には完全に利用不可能にしてしまう可能性がある 28。

EMNLP 2023で発表された研究では、動画キャプションにおける事実性の最初の人間評価が実施され、モデル生成された文の56%に事実誤りが含まれていることが判明した。これは、この分野における深刻な問題を示しているが、既存の評価指標は人間による事実性アノテーションとの相関がほとんどないことも明らかになった 28。例えば、既存の参照ベースの評価指標は、参照との類似性のみを測定するため、動画に実際には存在しない「幻覚」記述を含むキャプションを過小評価する可能性がある 30。この課題に対処するため、FactVCのような弱教師ありモデルベースの事実性評価指標が提案されており、動画キャプションの事実性評価において既存の指標よりもはるかに高い相関を達成している 28。また、EMScoreも「幻覚」キャプションを識別する可能性を示している 29。

「幻覚」問題は、モデルが訓練データにおける統計的な偏りや、視覚情報とテキスト情報の間の見かけ上の相関に過度に依存することで発生すると考えられる。例えば、特定のオブジェクトが特定の行動と頻繁に共起する場合、モデルは実際の視覚的な証拠がなくてもその行動を記述してしまう可能性がある 31。この問題は、キャプションの信頼性を損ない、特に医療、監視、自動運転などのクリティカルなアプリケーションでの採用を妨げるため、事実に基づいた正確なキャプション生成は引き続き重要な研究課題である。

### **3.3. 評価指標の限界と人間判断との乖離**

動画キャプションシステムの評価は、キャプションの流暢さ、単一シーン内で発生する複数のアクション、人間が動画で重要と考えることの推定など、複数の課題を考慮する必要がある困難なタスクである 32。ほとんどの指標は、システムが生成したキャプションが単一または一連の人間が生成したキャプションとどの程度類似しているかを測定することを目的としている 32。

しかし、機械翻訳や画像キャプションなどの他のタスクから借用されたBLEU、ROUGE、METEOR、CIDErといった広く適用されている自動評価指標は、動画キャプションの特殊な特性を無視する可能性があり、その発展を制限している 30。これらの指標は、参照がない動画には使用できないという本質的な欠点も抱えている 30。さらに、キャプションの一対多の性質により、正しいキャプションに対しても過剰なペナルティを与える可能性がある 30。人間による評価は最も理想的な指標であるが、時間と労力がかかる 30。このため、人間による判断との相関を最大化するために、人間による評価から直接学習する新しい自動指標が提案されている 32。BERTHAは、BERT言語モデルに基づいており、人間による評価と同様の評価を実行するように学習する 32。EMScoreは、参照不要な新しい指標であり、動画と候補キャプション間の類似性を直接測定する。これは、粗い粒度（動画とキャプション）と細かい粒度（フレームと単語）の両方のレベルでのマッチングスコアを組み合わせることで、動画の全体的な理解と詳細な特性を考慮する 30。EMScoreは、高い人間相関と低い参照依存性を示し、品質ドリフトに対しても堅牢であることが報告されている 30。評価指標の改善は、動画キャプション研究の進歩に不可欠である。既存の指標が人間による判断と乖離する問題は、モデルの真の性能を正確に評価し、研究の方向性を適切に導く上で大きな障壁となる。参照不要な指標や、動画コンテンツとキャプションの一貫性を直接評価する指標の開発は、この課題を克服し、より信頼性の高いモデル開発を促進する上で重要な役割を果たす。

### **3.4. バイアス（偏り）の緩和**

機械学習パイプラインのほぼすべての段階、すなわちタスク定義、データセット構築、テスト、デプロイメントにおいて、バイアスが生じる可能性がある 33。これにより、システムがユーザーに不十分なサービスを提供したり、すでに不利な立場にあるサブポピュレーションに不利益をもたらしたりする可能性がある 33。動画キャプション付与においても、データセットのバイアスやモデル出力のバイアスが問題となる。例えば、CLIPのような対照学習言語-画像事前学習モデルは、意図せずステレオタイプを吸収する可能性がある 34。これは、訓練データにおける表現バイアス（特定の属性への偏り）や関連バイアス（特定の概念と属性の関連付け）として現れる 34。

この課題に対処するため、データバランシング戦略が提案されている 34。また、Unbiasing through Textual Descriptions (UTD) という新しいスケーラブルな手法が提案されており、動画ベンチマークにおけるオブジェクトバイアスなどの表現バイアスを、自動的なテキスト記述を通じて分析・軽減する 35。UTDは、フレームごとのテキスト記述を生成し、そこからオブジェクト、活動、動詞などの概念をLLMを用いて抽出し、それらを用いてバイアスを分析する 35。これにより、既存の動画を修正することなく、バイアスを軽減した評価分割（UTD-splits）を作成し、より堅牢な動画理解能力の評価を可能にする 35。バイアスの緩和は、AIシステムの公平性と信頼性を確保するために不可欠である。特に、キャプションが社会的に重要な文脈（例：教育、ニュース）で使用される場合、偏った記述は誤解や差別を助長する可能性がある。データセットの収集段階から、モデルの学習、評価、デプロイメントに至るまで、バイアスを継続的に監視し、軽減する戦略が求められる。

### **3.5. 低リソース言語と文化的ニュアンスへの対応**

多言語動画キャプション付与は、グローバルな視聴者とのつながりを深める上で重要である 5。しかし、多言語環境での論理的推論は、言語間のミスマッチ、訓練データにおけるバイアス、低リソース言語におけるリソースの不足といった課題を伴う 37。

既存のLLMsやVLMは、主に高リソース言語に焦点を当てており、低リソース言語や類型的に異なる言語は十分に表現されていない 37。これにより、これらの言語では訓練データの利用可能性が限られ、パフォーマンスが低下し、言語理解や生成が効果的でなくなる可能性がある 38。さらに、あまり文書化されていない言語の微妙なニュアンスや文化的文脈を正確に捉えることの複雑さも、その利用をさらに困難にしている 38。この課題に対処するため、多言語モデルの研究が進められており、LLMsの能力を低リソース言語に拡張することを目指している 38。多言語動画プロジェクトを管理する上では、翻訳しやすいスクリプトの作成、異なるテキスト長に対応する視覚要素の計画、翻訳ブリーフの作成、読み取り速度に基づく字幕時間の調整、字幕テンプレートの使用、品質管理プロセスの確立などが推奨される 36。グローバル化が進む中で、多様な言語と文化に対応できるキャプション技術の需要は高まっている。単なる言語翻訳を超えて、文化的背景や文脈に即した適切な表現を生成する能力は、コンテンツの受容性を高め、より広範な視聴者層にリーチするために不可欠である。低リソース言語への対応は、デジタルデバイドを解消し、情報への普遍的なアクセスを促進する上で重要な社会的意義を持つ。

### **3.6. リアルタイム性・インタラクティブ性の向上**

動画キャプション付与の応用範囲を広げるためには、リアルタイムでのキャプション生成と、ユーザーがキャプションを調整できるインタラクティブな機能が不可欠である。特に、ライブコンテンツや即時性が求められる環境での利用において、この側面は重要性を増している。

聴覚障害者（DHH）学生の学習体験を向上させるためのリアルタイムARキャプションインターフェースに関する研究では、音声認識の進歩により、DHH個人にリアルタイムで話し言葉へのアクセスを提供する有望なソリューションとしてキャプションインターフェースが利用されていることが示されている 39。このシステムは、ユーザー中心設計（UCD）プロセスを通じて開発され、DHH参加者からの強い支持を得ている 39。これにより、学習体験が向上し、教育活動がサポートされることが示唆されている 39。ライブブロードキャストにおけるリアルタイムキャプションの管理には、ライブキャプション担当者がASR出力をサブ秒の遅延でタイプ、音声入力、または編集する必要がある 9。クラウドエディターを通じてASR出力をルーティングし、スピーカーIDを挿入したり、同音異義語を修正したりすることで、超低遅延ストリーミングプロトコルがキャプションをOTTプレーヤーに最小限のラグで配信する 9。リアルタイム性とインタラクティブ性の向上は、ライブイベント、オンライン会議、教育コンテンツ、人間-ロボットインタラクションなど、動的な環境でのキャプションの有用性を高める。特に、ARキャプションのような新しいインターフェースは、ユーザーの視線に直接キャプションを表示することで、没入感と理解度を向上させる可能性を秘めている。ユーザーフィードバックに基づいたシステム調整や、AIと人間の協調によるリアルタイム編集は、これらのシステムの精度と信頼性をさらに高める鍵となる。

## **4\. 今後の方向性と注目すべきトレンド（2025年7月現在）**

### **4.1. 法規制とアクセシビリティ要件の強化**

2025年に向けて、デジタルコンテンツのアクセシビリティへの注目が強まっており、米国ではADA（Americans with Disabilities Act）の更新、欧州では欧州アクセシビリティ法（EAA 2025）の施行がその主要な推進力となっている 5。EAA 2025は、テレビ放送やストリーミングサービスを含むすべての視聴覚メディアに、クローズドキャプションと字幕の統合を義務付けている 6。これにより、聴覚障害者や難聴者、およびテキストベースのサポートから恩恵を受ける視聴者がコンテンツに完全にアクセスできるようになる 6。

米国FCC（連邦通信委員会）も、2024年7月18日に閉鎖型キャプション設定に関する新規則を採択し、メーカーとMVPD（多チャンネルビデオ番組配信事業者）に対し、テレビ、ビデオストリーミングデバイス、特定のプリインストールアプリ上のキャプション表示設定を「容易にアクセス可能」にすることを義務付けている 42。これらの規則は、アクセシビリティを確保するための技術的要件だけでなく、ユーザー体験の向上にも焦点を当てている。これらの法規制の強化は、キャプション技術の研究開発と導入を強力に推進する外部要因となる。企業やコンテンツプロバイダーは、法的義務を果たすだけでなく、より広範な視聴者層にリーチし、ブランドロイヤルティを構築するために、高品質なキャプションサービスへの投資を増やすことが予想される 5。これにより、自動生成キャプションの精度向上、人間による編集との連携、多言語対応、およびユーザーカスタマイズ機能の強化が加速されると考えられる。

### **4.2. テキスト-動画生成（T2V）モデルとの連携**

制御可能なテキスト-動画生成（T2V）モデルは、テキストプロンプトを活用して動画合成をガイドし、デザインの即時視覚化やクリエイティブコンテンツ、エンターテイメントへの応用を促進する 43。動画キャプションは、T2V生成の重要な基盤インフラであり、粗粒度または詳細に欠けるキャプションは、視覚情報の理解と再構築を著しく妨げる 43。

このため、T2Vモデルでは、生成されたコンテンツと詳細なプロンプト/キャプションとのアライメントを強化することに注力している 43。VidCapBenchは、制御可能なT2V生成における動画キャプション評価のために特別に設計されたベンチマークであり、動画の美学、コンテンツ、動き、物理法則といった主要な側面を網羅している 43。このベンチマークは、専門家モデルによるラベリングと人間による洗練を組み合わせたデータアノテーションパイプラインを採用しており、T2Vモデルのトレーニングをガイドする上でその有用性が検証されている 43。動画キャプションは、T2Vモデルのトレーニングにおいて、セマンティックなアライメントを高める役割を果たす 43。T2V生成の進展は、キャプションが単に動画を記述するだけでなく、動画コンテンツの生成を制御するための「設計図」としての役割を担うことを意味する。これにより、キャプションの精度、詳細度、多様性、そして特定の側面（動き、オブジェクト、感情など）を制御する能力が、T2Vモデルの品質に直接影響を与えるため、キャプション研究はT2V分野と密接に連携しながら進化すると考えられる。

### **4.3. ユーザー意図に基づく制御可能なキャプション生成**

従来の動画キャプション付与は、動画の一般的な記述を生成することに焦点を当ててきたが、ACM Multimedia 2025のIntentVCチャレンジでは、ユーザー制御可能で意図指向の出力によって動画キャプションの根本的な限界に対処することを目指している 45。この課題は、特定のユーザー定義の意図に合わせてキャプションを調整することを可能にし、動画理解モジュールが動画の様々な側面を捉える能力をテストする 45。

MiraData、VDC、Vriptなどの他の手法も、被写体、背景、ショットなどの特定の側面に焦点を当てることで制御性を高め、T2V生成に大きく貢献している 43。また、イベントの観点から動画を記述し、時間情報をより効果的に捉える手法も存在する 43。ユーザー意図に基づく制御可能なキャプション生成は、キャプションのパーソナライゼーションと実用性を大幅に向上させる。例えば、教育用途では特定の概念に焦点を当てたキャプション、監視用途では異常行動に特化したキャプションなど、ユーザーの具体的なニーズに応じた情報を提供できるようになる。これは、キャプションが受動的な情報提供ツールから、能動的な情報探索・生成ツールへと進化する方向性を示しており、人間とAIの協調をより密接にする。

### **4.4. データ効率とモデル効率の最適化**

動画キャプションモデルは、一般的にリソース集約型であり、計算リソースが制約されている場合や、モデル設計とトレーニングの明確なガイダンスがない場合にパフォーマンスのボトルネックに直面することがある 26。このため、限られたリソースで高性能なキャプションを生成する手法の開発が重要視されている。

この課題に対処するため、リソース最適化の観点から、モデル規模の最適化、データ効率の最大化、強化学習の組み込みという3つの基本的な要因に焦点を当てた研究が進められている 26。例えば、事前学習済み画像-テキストモデルを動画キャプションに転用するアプローチは、わずか6,000組の動画-テキストペアという非常に少ないデータで、最先端の性能に匹敵する結果を達成している 26。これは、大規模なデータセットへの依存を減らし、低リソースシナリオでの実用的なソリューションを提供する 26。また、UGC-VideoCaptionerのようなモデルは、Gemini-2.5-Flashから蒸留された3Bパラメータのモデルであり、2段階の蒸留フレームワークを用いて、20,000のTikTok動画を自動ラベリングし、その後2,000の人間アラインされたキャプションでファインチューニングすることで、データ効率の高いソリューションを提供している 16。データ効率とモデル効率の最適化は、研究開発のコストを削減し、より多くの研究者や企業が動画キャプション技術に取り組むことを可能にする。特に、限られた計算リソースやデータしか持たない環境において、既存の事前学習済みモデルを効率的に転用する戦略や、合成データ生成の活用は、技術の実用化と普及を加速させる重要な要素となる。

## **5\. 結論と提言**

2025年7月現在、動画へのキャプション付与に関する研究は、技術的進歩と実世界応用の両面で急速な進化を遂げている。大規模マルチモーダルモデル（LMMs）の登場は、動画理解とキャプション生成の能力を飛躍的に向上させたが、長尺動画における詳細な記述生成や、生成されたキャプションの「幻覚」といった本質的な課題に直面している。これらの課題は、トレーニングデータの不足、既存評価指標の限界、およびモデルのバイアスに起因すると分析される。

しかしながら、これらの課題を克服するための革新的なアプローチも同時に進行している。合成データ生成フレームワークによる長尺キャプションデータの拡充、マルチモーダル融合技術の深化による音声・視覚情報の統合、拡散モデルの応用による生成品質の向上、圧縮ドメインからの直接キャプション生成による効率化、そして事前学習済み画像-テキストモデルの効率的な転用は、技術的ブレークスルーの兆候を示している。

さらに、欧州アクセシビリティ法（EAA 2025）や米国FCC規則といった法規制の強化は、アクセシブルなキャプションの普及を強力に後押しし、市場の需要を喚起している。テキスト-動画生成（T2V）モデルとの連携や、ユーザー意図に基づく制御可能なキャプション生成は、キャプションが単なる記述を超え、よりインタラクティブでパーソナライズされた体験を提供する未来を示唆している。

これらの分析に基づき、本報告書は以下の提言を行う。

* **データセットの拡充と多様化**: 長尺動画や低リソース言語、特定の文化的ニュアンスを反映した高品質なアノテーション付きデータセットの構築、および合成データ生成技術のさらなる開発が不可欠である。これにより、モデルの汎化能力と事実性が向上する。  
* **評価指標の革新**: 人間による判断とより高い相関を持つ、参照不要かつ事実性を評価できる新しい評価指標の開発を加速すべきである。これにより、モデルの真の性能を正確に測定し、研究開発の方向性を適切に導くことが可能となる。  
* **マルチモーダル統合の深化**: 視覚情報だけでなく、音声、非言語的キューなど、動画が持つすべてのモダリティを統合的に理解し、キャプションに反映させる研究を推進する。特に、ユーザー生成コンテンツ（UGC）などの実世界コンテンツにおけるオムニモーダル理解の強化が重要である。  
* **バイアス緩和の継続的な取り組み**: データ収集、モデル設計、評価の各段階でバイアスを特定し、軽減するための体系的な手法を確立する。公平で信頼性の高いキャプション生成は、社会的な受容性を高める上で極めて重要である。  
* **実用化に向けた効率化とインタラクティブ性の追求**: ストリーミング・リアルタイムキャプション技術、圧縮動画からの直接生成、および既存モデルの効率的な転用により、計算リソースの制約下でも高品質なキャプションを生成できる実用的なソリューションの開発を進める。また、ユーザーがキャプションを調整・制御できるインタラクティブなシステムの開発を通じて、多様なニーズに対応可能なキャプション体験を提供する。

これらの提言を実行することで、動画キャプション付与技術は、情報アクセスを民主化し、コンテンツの価値を最大化する上で、より中心的かつ不可欠な役割を果たすことができると結論付けられる。

#### **引用文献**

1. Build a Video Search and Summarization Agent with NVIDIA AI Blueprint, 7月 24, 2025にアクセス、 [https://developer.nvidia.com/blog/build-a-video-search-and-summarization-agent-with-nvidia-ai-blueprint/](https://developer.nvidia.com/blog/build-a-video-search-and-summarization-agent-with-nvidia-ai-blueprint/)  
2. Video Captioning Based on Both Egocentric and Exocentric Views of Robot Vision for Human-Robot Interaction \- ResearchGate, 7月 24, 2025にアクセス、 [https://www.researchgate.net/publication/356656590\_Video\_Captioning\_Based\_on\_Both\_Egocentric\_and\_Exocentric\_Views\_of\_Robot\_Vision\_for\_Human-Robot\_Interaction](https://www.researchgate.net/publication/356656590_Video_Captioning_Based_on_Both_Egocentric_and_Exocentric_Views_of_Robot_Vision_for_Human-Robot_Interaction)  
3. Video Captioning | Papers With Code, 7月 24, 2025にアクセス、 [https://paperswithcode.com/task/video-captioning](https://paperswithcode.com/task/video-captioning)  
4. State of Video Report: Video Marketing Statistics for 2025 \- Wistia Blog, 7月 24, 2025にアクセス、 [https://wistia.com/learn/marketing/video-marketing-statistics](https://wistia.com/learn/marketing/video-marketing-statistics)  
5. The importance of subtitles in your videos in 2025 – LiLiCAST, 7月 24, 2025にアクセス、 [https://lilicast.com/importance-of-subtitles-in-2025/](https://lilicast.com/importance-of-subtitles-in-2025/)  
6. CaptionHub Blog | The European Accessibility Act 2025 | CaptionHub, 7月 24, 2025にアクセス、 [https://www.captionhub.com/blog-post/eaa2025/](https://www.captionhub.com/blog-post/eaa2025/)  
7. Video and Audio | Office for Digital Accessibility (ODA), 7月 24, 2025にアクセス、 [https://accessibility.umn.edu/getting-started/learn-7-core-accessibility-skills/video-audio](https://accessibility.umn.edu/getting-started/learn-7-core-accessibility-skills/video-audio)  
8. Captions/Subtitles | Web Accessibility Initiative (WAI) | W3C, 7月 24, 2025にアクセス、 [https://www.w3.org/WAI/media/av/captions/](https://www.w3.org/WAI/media/av/captions/)  
9. Multilingual Captioning for Broadcasters \- Digital Nirvana, 7月 24, 2025にアクセス、 [https://digital-nirvana.com/blog/multilingual-captioning-broadcast/](https://digital-nirvana.com/blog/multilingual-captioning-broadcast/)  
10. LongCaptioning: Unlocking the Power of Long Video Caption Generation in Large Multimodal Models \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/html/2502.15393v2](https://arxiv.org/html/2502.15393v2)  
11. LongCaptioning: Unlocking the Power of Long Caption Generation in Large Multimodal Models \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/html/2502.15393v1](https://arxiv.org/html/2502.15393v1)  
12. Dense Video Captioning | Papers With Code, 7月 24, 2025にアクセス、 [https://paperswithcode.com/task/dense-video-captioning/codeless](https://paperswithcode.com/task/dense-video-captioning/codeless)  
13. Show Think and Tell: Thought-Augmented Fine ... \- CVF Open Access, 7月 24, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Kim\_Show\_Think\_and\_Tell\_Thought-Augmented\_Fine-Tuning\_of\_Large\_Language\_Models\_CVPRW\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Kim_Show_Think_and_Tell_Thought-Augmented_Fine-Tuning_of_Large_Language_Models_CVPRW_2024_paper.pdf)  
14. Multi-Task Video Captioning with a Stepwise Multimodal Encoder \- MDPI, 7月 24, 2025にアクセス、 [https://www.mdpi.com/2079-9292/11/17/2639](https://www.mdpi.com/2079-9292/11/17/2639)  
15. End-to-End Generative Pretraining for ... \- CVF Open Access, 7月 24, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2022/papers/Seo\_End-to-End\_Generative\_Pretraining\_for\_Multimodal\_Video\_Captioning\_CVPR\_2022\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Seo_End-to-End_Generative_Pretraining_for_Multimodal_Video_Captioning_CVPR_2022_paper.pdf)  
16. UGC-VideoCaptioner: An Omni UGC Video Detail Caption Model and New Benchmarks \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/html/2507.11336v1](https://arxiv.org/html/2507.11336v1)  
17. (PDF) UGC-VideoCaptioner: An Omni UGC Video Detail Caption Model and New Benchmarks \- ResearchGate, 7月 24, 2025にアクセス、 [https://www.researchgate.net/publication/393723312\_UGC-VideoCaptioner\_An\_Omni\_UGC\_Video\_Detail\_Caption\_Model\_and\_New\_Benchmarks](https://www.researchgate.net/publication/393723312_UGC-VideoCaptioner_An_Omni_UGC_Video_Detail_Caption_Model_and_New_Benchmarks)  
18. (PDF) Vision-Text Cross-Modal Fusion for Accurate Video Captioning \- ResearchGate, 7月 24, 2025にアクセス、 [https://www.researchgate.net/publication/374686555\_Vision-text\_cross-modal\_fusion\_for\_accurate\_video\_captioning](https://www.researchgate.net/publication/374686555_Vision-text_cross-modal_fusion_for_accurate_video_captioning)  
19. \[2201.08264\] End-to-end Generative Pretraining for Multimodal Video Captioning \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/abs/2201.08264](https://arxiv.org/abs/2201.08264)  
20. Diffusion-Based Multimodal Video Captioning | OpenReview, 7月 24, 2025にアクセス、 [https://openreview.net/forum?id=iDpuX9wwWI](https://openreview.net/forum?id=iDpuX9wwWI)  
21. Diffusion-Based Multimodal Video Captioning \- Aalto University's research portal, 7月 24, 2025にアクセス、 [https://research.aalto.fi/en/publications/diffusion-based-multimodal-video-captioning](https://research.aalto.fi/en/publications/diffusion-based-multimodal-video-captioning)  
22. End-to-end Dense Video Captioning as Sequence Generation \- ACL Anthology, 7月 24, 2025にアクセス、 [https://aclanthology.org/2022.coling-1.498.pdf](https://aclanthology.org/2022.coling-1.498.pdf)  
23. Streaming Dense Video Captioning, 7月 24, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2024/html/Zhou\_Streaming\_Dense\_Video\_Captioning\_CVPR\_2024\_paper.html](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Streaming_Dense_Video_Captioning_CVPR_2024_paper.html)  
24. CVPR Poster Streaming Dense Video Captioning, 7月 24, 2025にアクセス、 [https://cvpr.thecvf.com/virtual/2024/poster/31433](https://cvpr.thecvf.com/virtual/2024/poster/31433)  
25. ICCV 2023 Open Access Repository, 7月 24, 2025にアクセス、 [https://openaccess.thecvf.com/content/ICCV2023/html/Shen\_Accurate\_and\_Fast\_Compressed\_Video\_Captioning\_ICCV\_2023\_paper.html](https://openaccess.thecvf.com/content/ICCV2023/html/Shen_Accurate_and_Fast_Compressed_Video_Captioning_ICCV_2023_paper.html)  
26. Pretrained Image-Text Models are Secretly Video Captioners \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/html/2502.13363v1](https://arxiv.org/html/2502.13363v1)  
27. \[2502.13363\] Pretrained Image-Text Models are Secretly Video Captioners \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/abs/2502.13363](https://arxiv.org/abs/2502.13363)  
28. Models See Hallucinations: Evaluating the Factuality in Video Captioning \- ACL Anthology, 7月 24, 2025にアクセス、 [https://aclanthology.org/2023.emnlp-main.723/](https://aclanthology.org/2023.emnlp-main.723/)  
29. Models See Hallucinations: Evaluating the Factuality in Video Captioning \- ACL Anthology, 7月 24, 2025にアクセス、 [https://aclanthology.org/2023.emnlp-main.723.pdf](https://aclanthology.org/2023.emnlp-main.723.pdf)  
30. EMScore: Evaluating Video Captioning via Coarse-Grained and Fine-Grained Embedding Matching \- CVF Open Access, 7月 24, 2025にアクセス、 [https://openaccess.thecvf.com/content/CVPR2022/papers/Shi\_EMScore\_Evaluating\_Video\_Captioning\_via\_Coarse-Grained\_and\_Fine-Grained\_Embedding\_Matching\_CVPR\_2022\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_EMScore_Evaluating_Video_Captioning_via_Coarse-Grained_and_Fine-Grained_Embedding_Matching_CVPR_2022_paper.pdf)  
31. Mitigating Dataset Bias in Image Captioning Through Clip Confounder-Free Captioning Network \- ResearchGate, 7月 24, 2025にアクセス、 [https://www.researchgate.net/publication/374543465\_Mitigating\_Dataset\_Bias\_in\_Image\_Captioning\_Through\_Clip\_Confounder-Free\_Captioning\_Network](https://www.researchgate.net/publication/374543465_Mitigating_Dataset_Bias_in_Image_Captioning_Through_Clip_Confounder-Free_Captioning_Network)  
32. BERTHA: Video Captioning Evaluation Via Transfer-Learned Human Assessment \- ACL Anthology, 7月 24, 2025にアクセス、 [https://aclanthology.org/2022.lrec-1.168.pdf](https://aclanthology.org/2022.lrec-1.168.pdf)  
33. Machine Learning and Fairness \- Microsoft Research, 7月 24, 2025にアクセス、 [https://www.microsoft.com/en-us/research/video/machine-learning-and-fairness/](https://www.microsoft.com/en-us/research/video/machine-learning-and-fairness/)  
34. CLIP the Bias: How Useful is Balancing Data in Multimodal Learning? \- OpenReview, 7月 24, 2025にアクセス、 [https://openreview.net/forum?id=FIGXAxr9E4](https://openreview.net/forum?id=FIGXAxr9E4)  
35. Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/html/2503.18637v1](https://arxiv.org/html/2503.18637v1)  
36. Manage Multilingual Video Projects to Expand Your Global Audience \- Content Beta, 7月 24, 2025にアクセス、 [https://www.contentbeta.com/blog/manage-multilingual-video/](https://www.contentbeta.com/blog/manage-multilingual-video/)  
37. A Survey of Multilingual Reasoning in Language Models \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/html/2502.09457v1](https://arxiv.org/html/2502.09457v1)  
38. Foundation Models for Low-Resource Language Education (Vision Paper) \- Qeios, 7月 24, 2025にアクセス、 [https://www.qeios.com/read/IQU339](https://www.qeios.com/read/IQU339)  
39. (PDF) Tailored Real-time AR Captioning Interface for Enhancing Learning Experience of Deaf and Hard-of-Hearing (DHH) Students \- ResearchGate, 7月 24, 2025にアクセス、 [https://www.researchgate.net/publication/387766707\_Tailored\_Real-time\_AR\_Captioning\_Interface\_for\_Enhancing\_Learning\_Experience\_of\_Deaf\_and\_Hard-of-Hearing\_DHH\_Students](https://www.researchgate.net/publication/387766707_Tailored_Real-time_AR_Captioning_Interface_for_Enhancing_Learning_Experience_of_Deaf_and_Hard-of-Hearing_DHH_Students)  
40. Tailored Real-time AR Captioning Interface for Enhancing Learning Experience of Deaf and Hard-of-Hearing (DHH) Students \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/html/2501.02233v1](https://arxiv.org/html/2501.02233v1)  
41. Accessibility Standards in 2025: The Essential Role of Human Captioning \- CaptionLabs, 7月 24, 2025にアクセス、 [https://captionlabs.com/blog/accessibility-standards-2025-human-captioning/](https://captionlabs.com/blog/accessibility-standards-2025-human-captioning/)  
42. FCC Adopts New Accessibility Rules for Closed Captioning Settings ..., 7月 24, 2025にアクセス、 [https://www.dwt.com/blogs/broadband-advisor/2024/07/fcc-rules-on-closed-captioning-accessibility](https://www.dwt.com/blogs/broadband-advisor/2024/07/fcc-rules-on-closed-captioning-accessibility)  
43. VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation \- arXiv, 7月 24, 2025にアクセス、 [https://arxiv.org/html/2502.12782v1](https://arxiv.org/html/2502.12782v1)  
44. VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation \- Powerdrill, 7月 24, 2025にアクセス、 [https://powerdrill.ai/discover/summary-vidcapbench-a-comprehensive-benchmark-of-video-cm7cefmekk1z107t7auvrhlh8](https://powerdrill.ai/discover/summary-vidcapbench-a-comprehensive-benchmark-of-video-cm7cefmekk1z107t7auvrhlh8)  
45. Grand Challenges | 2025 ACM Multimedia \- ACM MM 2025, 7月 24, 2025にアクセス、 [https://acmmm2025.org/grand-challenge/](https://acmmm2025.org/grand-challenge/)
