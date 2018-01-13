学習用プログラム

【フォルダ】
/lamg_m/：Julus用の単語辞書ファイル（日本語音節辞書など）が入っているフォルダ


【ファイル】
CNN_place.py：CNNの特徴量をそのまま保存する.(PlaceCNN版)
Julius1best_gmm.py：主にPRR評価用に使用。Juliusで音声認識し、n-best音声認識結果を得る
JuliusLattice_gmm.py：学習プログラムと連携して、Juliusで音声認識しWFST音声認識結果を得る
README.txt：このファイル
SpCoSLAM.sh：オンライン学習実行用基本ファイル
__init__.py：ハイパーパラメータなどの初期値やファイルパス指定用のファイル
autovisualization.py：自動で学習結果を逐次描画するためのプログラム(保存は別でしなければならない)
collectmapclocktime.py：動画作成のために、地図の時刻をひとつのファイルにまとめて生成するためのもの。
gmapping.sh：FastSLAM実行用のシェルスクリプト
learnSpCoSLAM3.2.py：SpCoSLAM online learning program (無駄なコメントアウトコードを省いたバージョン+bugfix)、かつ、SpCoAのオンライン版（SpCoSLAMから言語モデル更新と画像特徴を除けるバージョン）
map_saver.py：地図を逐次的に保存していくプログラム（rospy使用）
new_place_draw.py：学習した場所領域のサンプルをrviz上に可視化するプログラム(石伏（サンプルプロット）→磯辺（ガウス概形）→彰)
new_place_draw_online.py：オンライン可視化用
new_position_draw_online.py：ロボットの自己位置描画用
run_SpCoSLAM.py：SpCoSLAMを実行するための子プログラム
run_gmapping.sh：gmappingを実行するための子プログラム
run_mapviewer.py：map_serverコマンドを実行する小プログラム（未使用？）
run_mapviewer.sh：run_mapviewer.pyを実行するためのシェルスクリプト（未使用？）
run_rosbag.py：rosbagを実行するための子プログラム
run_roscore.sh：roscoreを実行するための子プログラム
saveSpCoMAP.rviz：rvizファイル
saveSpCoMAP_online.rviz：rvizファイル（オンライン描画用）



【実行準備】
・学習データセットのパス指定、トピック名を合わせるなど（__init__.py、run_gmapping.sh）
・学習データセットの時刻情報より、教示時刻を保存したファイルを作成（）
・音声データファイルを用意しておく。__init__.pyでファイルパス指定。
・学習プログラム実行の前に、CNN_place.pyを実行。画像特徴のファイルフォルダを作成しておく。
・パーティクル数の指定は、__init__.pyとrun_gmapping.shの両方変更する必要がある。

【実行方法】
cd ~/Dropbox/SpCoSLAM/learning
./SpCoSLAM.sh

【注意事項】
・run_rosbag.pyでたまにgflag関係のエラーがでることがあるが、ファイル読み込み失敗が原因。callback関数なのでほっておけば再読み込みしてくれて、動くので問題ない。
・低スペックPCでは、gmappingの処理が追いつかずに地図がうまくできないことがある。

-----
[rviz上に位置分布を描画する方法]
Googleドライブのファイル参照。
roscore
rviz -d ./Dropbox/SpCoSLAM/learning/saveSpCoMAP_online.rviz 
python ./autovisualization.py p30a20g10sfix008

個別指定の場合
rosrun map_server map_server ./p30a20g10sfix008/map/map361.yaml
python ./new_place_draw.py p30a20g10sfix008 50 23 

-------------------------------------------------
更新日時
2017/02/12 Akira Taniguchi
2017/03/12 Akira Taniguchi
2018/01/12 Akira Taniguchi