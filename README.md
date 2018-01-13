# SpCoSLAM

SpCoSLAMの実装 (オンライン場所概念獲得、地図生成、語彙獲得)
IROS2017の実験において使用したプログラム

We propose an online learning algorithm based on a Rao-Blackwellized particle filter for spatial concept acquisition and mapping. We have proposed a nonparametric Bayesian spatial concept acquisition model (SpCoA). We propose a novel method (SpCoSLAM) integrating SpCoA and FastSLAM in the theoretical framework of the Bayesian generative model. The proposed method can simultaneously learn place categories and lexicons while incrementally generating an environmental map. 

【実行環境】
Ubuntu　14.04
Pythonバージョン2.7.6
ROS indigo
Caffe (リファレンスモデル:Places-205)
Julius dictation-kit-v4.3.1-linux (日本語音節辞書使用、ラティス出力)
語彙獲得ありの場合：latticelm 0.4, OpenFST

IROS2017では、albert-B-laser-vision-datasetのrosbagファイルを使用

【実行準備】
・学習データセットのパス指定、トピック名を合わせるなど（__init__.py、run_gmapping.sh）
・学習データセットの時刻情報より、教示時刻を保存したファイルを作成
・音声データファイルを用意しておく。__init__.pyでファイルパス指定。
・学習プログラム実行の前に、CNN_place.pyを実行。画像特徴のファイルフォルダを作成しておく。
・パーティクル数の指定は、__init__.pyとrun_gmapping.shの両方変更する必要がある。

【実行方法】
cd ~/SpCoSLAM/learning
./SpCoSLAM.sh

【注意事項】
・run_rosbag.pyでたまにgflag関係のエラーがでることがあるが、ファイル読み込み失敗が原因。ほっておけば再読み込みしてくれて、動くので問題ない。
・低スペックPCでは、gmappingの処理が追いつかずに地図がうまくできないことがある。


このリポジトリにはgmappingが含まれます。
gmappingはオリジナルバージョンのライセンスに従います。

---
このプログラムを使用したものを公開される場合は、必ず引用情報を明記してください。

Reference:
Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS2017), 2017.


Original paper:
https://arxiv.org/abs/1704.04664

Sample video:
https://youtu.be/z73iqwKL-Qk

2018/01/15  Akira Taniguchi
