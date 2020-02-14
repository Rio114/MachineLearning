= 二値分類モデルの評価

== 二値分類モデル

=== 二値分類モデル

二値分類モデルは様々な形で構成できる。例えば、あるデータ点に対して対象となる事象発生の予測確率の大きさを示す値が得られるとき、適当な閾値で集団を分けて閾値より大きな予測値あのものは正、閾値より小さい予測値のものを負と予測することで二値分類を構成できる。また、ルールベースによりデータの持つ属性がある条件を満たすときに正と予測するとしても二値分類を構成できる。

後者はより一般的な場合であり条件ごとに混合行列を検証しなくてはならないが、前者は条件が予測値という一次元の数値で表されているので閾値を一次元的に動かすだけで連続的に評価できる。

ここでは前者の評価方法を整理する。この場合、閾値を決めることで各データ点が混合行列のどこに入るかが決まる。しかし、これではPrecisionやRecallなどの指標は閾値次第で分類の結果が変わってしまうため、複数のモデルを比較できるように閾値に依存しない形でモデルの評価を行いたい。

そこでよく用いられるのがROC曲線やPR曲線、そしてそれらをそれぞれ一つの数字に落とし込んだものがAUCやAPである。なお、ROC曲線やPR曲線を作成するときには、モデルの予測値は確率のように0から1の範囲である必要はないが、以下では簡単のためモデルの予測値は0から1の範囲の予測確率を示すものとする。

=== 良いモデルとは

そもそも良いモデル、悪いモデルというのはどういうものなのだろうか。一旦はあまり深く考えずに以下のようなモデルがありそうだということが想像できる。

 * 神モデル：森羅万象を司る神様のように正例に対しては1、負例に対しては0を出力して、正解率100%を達成する、事象の発生を確実に予言できるモデル。

 * ポンコツモデル：正解ラベルがなんであっても二値分類スコアとして0から1の一様乱数を出力するモデル。予測には全く使えない。

 * 平凡モデル：正例ならば1に近い予測値、負例ならば0に近い予測値を返す。同じ予測値でも正解には正例と負例が混ざってる。神ほどはっきり予測はできていないが、ポンコツよりは有益な情報を得られる。

 * 天邪鬼モデル：正例に対しては低い予測値、負例に対しては高い予測値を返すモデル。予測を反転させれば平凡なモデルになるため、実質的に平凡モデルである。

確実ではないがそこそこな予測ができる平凡なモデルが複数あったとして、それらを比較するためにはその良し悪しを数値で表現したい。
そこでよく用いられるのがAUCという指標である。
AUCがどんなものであるかは後ほど説明するとして、忙しい人はAUCが0.8を超えると良いモデルで0.9を超えると過学習だと短絡的に判断してしまうことがよくある。
しかし、ビッグで不均衡なデータだと多様性により学習が困難となり、AUCが0.7に満たないモデルで運用していることもある。
結局、どんな精度指標でも単体では意味をなさず、必ず何を重視するのかを決めたうえで、相対比較を行っていくことが客観的な評価となる。

=== 閾値による二値分類
正例であることを予測する値、予測値を出力する二値分類モデルでは、予測値が高いほど正例である可能性が高いことが期待される。
そこで予測値の順番にデータを並べて、ある閾値によってデータを二分割して予測値が高い集団を正、低い集団を負と予測することで二値分類を行うことができる。
二値分類のトレードオフは誤検知（FP）と見逃し（FN）であることを思い出すと、この二つをプロットすることにより、二値分類モデルの性質を可視化できそうである。
慣例に従って、見逃し（FN）をRecallで評価すると誤検知は以下のように二通りの見方がある。

 * FPが負例のうちの何割か。FPRで誤検知を評価する。
 * FPが正の予測のうちの何割か。Precisionで誤検知を評価する。

前者はROC曲線といい、後者はPR曲線という。
二つあってややこしいが、両方とも正例の捕捉率に対する誤検知（FP）を評価している点で共通している。

== ROC曲線
ROC曲線では見逃しをTPR（Recall）、負例が誤って正と予測されたFPRで誤検知を評価する。
通常、負例の中から正と予測した割合（FPR）よりも正例の中から正と予測した割合（TPR）のほうが大きくなる。
もし逆にFPRが大きい場合、正の予測を負の予測だと読み替えることで負例を効率的に集めることができる。
閾値を連続的に変えていったときに、FPRとTPRの組み合わせがどのように変化していくかを視覚化したのがReceiver Operating Characteristic curve（受信者動作特性曲線)、略してROC曲線である。

=== 曲線の描き方
予測値（Score）の順番にデータ点を並べて、閾値を動かしていった時に、x軸にFPR、y軸にTPRを取ったものがROC曲線である。
@<table>{thres06}、@<table>{thres04}は予測値、正例負例のflgも同一であるが、閾値だけが異なるためNo.5, 6の混合行列のクラスが異なっている。
それに伴って、FPRとTPRの組み合わせも異なっている。

//table[thres06][スコア0.6以上は正予測]{
No.	Score	flg	class
------------
1	0.9	1	TP
2	0.8	1	TP
3	0.7	0	FP
4	0.6	1	TP
:	（閾値）	:	:
5	0.5	1	FN
6	0.4	0	TN
7	0.3	1	FN
8	0.2	0	TN
9	0.1	0	TN
//}

//list[roc06][スコア0.6以上を正予測とするときのFPR, TPR]{
FPR = FP数 / （flg=0の数） = 1 / 5 = 0.2
TPR = TP数 / （flg=1の数） = 3 / 5 = 0.6
//}

//table[thres04][スコア0.4以上は正予測]{
No.	Score	flg	class
------------
1	0.9	1	TP
2	0.8	1	TP
3	0.7	0	FP
4	0.6	1	TP
5	0.5	1	TP
6	0.4	0	FP
:	（閾値）	:	:
7	0.3	1	FN
8	0.2	0	TN
9	0.1	0	TN
//}

//list[roc04][スコア0.4以上を正予測とするときのFPR, TPR]{
FPR = FP数 / （flg=0の数） = 2 / 5 = 0.4
TPR = TP数 / （flg=1の数） = 4 / 5 = 0.8
//}

同様の表とFPRとTPRの組み合わせを予測値の値の数だけ作ることができる。
正確に言うと、閾値の数は予測値の数よりも１少ない。
上の表の例でいうと、予測値は0.9から0.1まで0.1刻みで9個の値があるので、0.9以上、、、0.2以上というように8個の閾値を設定できる。
従って、上記の例ではROC曲線は8個の点を結んだものになる。

@<img>{conf_mat_unbalance}は@<table>{thres06}、@<table>{thres04}から算出される点（FPR, TPR）をプロットした例である。

//image[roc][ROC曲線][scale=0.7]{
//}

=== 曲線の特徴
でたらめなスコアを出力するポンコツモデルであれば、正例と負例でスコアの分布が同じものになり、どんな閾値でもFPRもTPRも等しくなる。
つまり、ポンコツモデルのROCは斜め45度の線になる。

正例を全て捕捉するために多くのデータを正であると予測すると、その正予測の中には多くの負例が混入することになり、FPRが高くなる。
また、負例をすべて間違いなく負と予測したければ、データ全体を負と予測すれば達成できるが、その時は正例を全く捕捉できない。

@<img>{conf_mat_unbalance}では、少なく正予測を行ったときと多めに正予測をした時の混合行列をイメージしたものである。マスの大きさが分類されたデータの数を示している。
正例と負例では集団が異なるので同じ閾値を用いても正と予測される割合は異なるが、二値分類モデルが天邪鬼でない限り、正例の内の正予測の割合が大きくなる。つまり、FPRよりTPRが大きい。
また、多めに正の予測を行うと、相応に負例を正と予測する割合も増える。
以上のことから、ROC曲線は斜め45度線より上に弓なりの形状となる。

//image[conf_mat_unbalance][【上段】厳選して正予測を行ったときの混合行列、【下段】多めに正予測を行ったときの混合行列）][scale=0.5]{
//}

なお、冒頭でも述べたように予測値はは順番に並べることができれば確率値のように0から1の範囲である必要はない。上記の表の例でいうと、予測値を一律2倍にしてもデータの順番は変わらない。さらに言えば、非線形な変換であっても順序が保たれていれば、ROC曲線は不変である。逆に言うと、ROC曲線は予測値の順序が正例負例の割合を反映しているかを読み取ることができるものの、絶対値に関する情報を読み取ることができない。

=== AUC
ROCの下側面積Area Under Curveを略してAUCと言い、モデルの良し悪しを一つの数値で表すことができる。これはTPR、すなわちRecallの平均と見ることができるので、モデルによる正例の捕捉をどれだけ期待できるかを示している。

ポンコツなモデルだとAUCは0.5になる。つまり、閾値を色々変えて平均してみると半分くらいは正例を捕捉できるということである。
また、一つの閾値で正負を分けることができればAUCは1になる。

AUCが同じなら左のほうの立ち上がりが良いほうが良いモデルである。なぜならその場合、FPRをあまり上げずにTPRを上げることができているからである。

=== 使用例
正例、負例ともに全体の数に対してそれぞれ何割が正と予測されたかを図示したのがROC曲線である。
正例を当てることは当然であるが、負例を正しく負と予測したい場合にROC曲線は有効である。
例えば疾患の検査で、罹患していない人に対して誤診断による必要以上の精密検査は損失が大きいといった状況で、正例の捕捉と負例の捕捉のバランスを考えるときなどで有効である。

== Precision-Recall曲線
見逃しをRecallで評価し、正予測に対して誤って正と予測した割合Precisionで誤検知を評価したいときにはPrecision-Recall曲線を用いる。

つまり、正例を多く捕捉しつつ、正の予測ではなるべく多くを言い当てたい。ROC曲線とのニュアンスの違いが難しいが、こちらでは負例の数についてはあまり興味がない状況を念頭に置いている。これにより、正例の捕捉割合を上げることで正の予測の正しさがどれだけ下がるかのトレードオフを視覚化できる。

=== 曲線の描き方
スコアの順番にデータ点を並べて、閾値を動かしていった時に、x軸にRecall、y軸にPrecisionを取ったものがPrecision-Recall曲線である。ROC曲線の書き方で挙げた例で言うと、以下のように計算される。

//list[rp06][TPR, FPRの計算例]{
Recall = TP数 / （flg=1の数） = 3 / 5 = 0.6
Precision = TP数 / （TP+FPの数） = 3 / 4 = 0.75
//}

//list[rp04][TPR, FPRの計算例]{
Recall = TP数 / （flg=1の数） = 4 / 5 = 0.8
Precision = TP数 / （TP+FPの数） = 4 / 6 = 0.66
//}

=== AP（PR曲線のAUC）
PR曲線でもAUCを考えることができる。これはROC曲線のAUCと区別してAverage Precision、略してAPという。名前の通り、APはモデルによる正例の言い当て率がどれくらい期待できるかということを示している。

ポンコツなモデルでは、モデルで予測を行う前の集団の正例の割合となる。
APだけだと集団間での比較が難しいとき、APが元の正例割合の何倍になるかという尺度、つまり平均Liftにすると比較がしやすくなる。
また、一つの閾値で正負を分けることができればAPは1になる。

=== 曲線の特徴
正例を多く捉えるために閾値を緩くすることで、負例を誤って正としてしまう（FP）数が増えるので通常右肩下がりの曲線となる。ただし、特に不均衡データの場合、左のほうでPrecisionの分母となる正の予測数（TP+FP）が少ないため変動が大きい。逆に言うと、対象の集団の中でも予測値が大きく正予測されやすい集団の影響を大きく受ける。

PR曲線もROC曲線と同様に、予測値は順番に並べることができれば確率値のように0から1の範囲である必要はない。逆に言うと、PR曲線は予測値の順序性を反映しているものの、絶対値に関する情報を読み取ることができない。

=== 使用例
正例の捕捉率に対して、正の予測の中にどれだけ誤検知が混ざっているかということに着目したのがPR曲線である。
予測値が高く、正と予測されやすい集団を重視する場合に有効である。
例えば、あるサービス訴求を行いたいが、予算の関係で対象者を絞らなくてはならないといった場面である。こうしたとき、正予測の中の無駄打ちをなくすためにはPrecisionが高い予測を行う必要がある。

== 二値分類モデルの例
これまでの述べてきたモデル評価を、実際のデータから学習したモデルについて行ってみたい。

=== データセット：Santander
二値分類の例としてKaggleからSantander Customer Transaction Predictionのデータを用いた。

https://www.kaggle.com/c/santander-customer-transaction-prediction

これは顧客属性から金融商品の購入予測を行うものである。
説明変数である顧客属性は200種類の連続変数であり、どんな金融商品かは述べられていないが目的変数は0か1の二値である。
また、目的変数のうち正例は10%であるため、不均衡データとしても手ごろな例だと言える。
評価指標はROC曲線のAUCを用いる。

=== アルゴリズム：LightGBM
学習の例としてここではLightGBMを用いた。
LightGBMを詳しく説明すると本書の目的を逸脱してしまうので、アルゴリズムの次に簡単に述べるに留める。
簡単に言うと、弱分類器である決定木を何本も育ててアンサンブルしていく過程で勾配ブースティングにより一本前の木の誤りを重点的に学習して、全体として高性能の分類器をつくるというものである。
非常に強力な手法であり、Kaggleでも人気のアルゴリズムである。

今回パラメータは@<table>{lgbm_params}を用いている。
表にないものは規定値である。

//table[lgbm_params][LightGBMで用いたパラメータ]{
名称	値
------------
objective	binary
num_leaves	15
is_unbalance	True
num_boost_round	100
//}

=== モデル評価
==== ROC曲線
PR曲線は@<img>{roc_plot}のようになった。
AUCはTrainの0.91に対してValidでは0.86となった。

//image[roc_plot][ROC曲線][scale=0.7]{
//}

==== PR曲線
PR曲線は@<img>{pr_plot}のようになった。APはTrainの0.62に対してValidでは0.50。また、元の割合で除した平均LiftはTrainの6.18に対してValidでは5.01となった。

//image[pr_plot][Recall_Precision曲線][scale=0.7]{
//}

==== 両者の比較
同じモデルを評価しているのでROC曲線、PR曲線共に同じことが読み取れる。

 * TrainとValidで差異があり過学習している
 * Trainと同じだけのTPR（Recall）で正例を捕捉しようとするとValidではFPがより多くなる。

さらに分析を進める上で、ROC曲線とPR曲線のどちらを用いて議論をしていくかは、具体的な目的次第である。
サービス加入の可能性が高いと予測した中での無駄打ちを20%程度に抑えたいので、本当にサービス加入してくれるであろう顧客の80%の見逃しには目を瞑るという話ならPR曲線を見ながら議論できる。
また、見込みのない人への訴求は10%に抑えたいので、本当にサービス加入してくれるであろう顧客の40%の見逃すのも止む無し、という議論であればROC曲線で見る。
