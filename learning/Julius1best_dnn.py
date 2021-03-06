# encoding: shift_jis
# Akira Taniguchi 2017/02/25-2017/06/28-2071/07/06
# 実行すれば、自動的に指定フォルダ内にある音声ファイルを読み込み、Juliusで通常の音声認識した結果をN-bestで出力してくれる。
# 注意点：指定フォルダ名があっているか確認すること。
import glob
import codecs
import os
import re
import sys
from __init__ import *

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass


# juliusに入力するためのwavファイルのリストファイルを作成
def MakeTmpWavListFile( wavfile , trialname):
    Makedir( datafolder + trialname + "/" + "tmp" )
    Makedir( datafolder + trialname + "/" + "tmp/" + trialname )
    fList = codecs.open( datafolder + trialname + "/" + "tmp/" + trialname + "/list.txt" , "w" , "sjis" )
    fList.write( wavfile )
    fList.close()


# Lattice認識
def RecogLattice( wavfile , step , filename , trialname, nbest):
    MakeTmpWavListFile( wavfile , trialname )
    if (JuliusVer == "v4.4"):
      binfolder = "bin/linux/julius"
    else:
      binfolder = "bin/julius"
    if (step == 0 or step == 1):  #最初は日本語音節のみの単語辞書を使用(step==1が最初)####使用されないはず
      p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-dnn.jconf -v " + lmfolder + lang_init_DNN + " -demo -filelist "+ datafolder + trialname + "/tmp/" + trialname + "/list.txt -output " + str(NbestNum+1) + " -dnnconf " + Juliusfolder + "julius.dnnconf $*" ) #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS 
      print "Read dic:" ,lang_init_DNN , step
    else:  #更新した単語辞書を使用
      #print Juliusfolder + "bin/julius -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + datafolder + trialname + "/" + str(step-1) + "/WD.htkdic -demo -filelist tmp/" + trialname + "/list.txt -confnet -lattice"
      p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-dnn.jconf -v " + datafolder + trialname + "/" + str(step-1) + "/WD.htkdic -demo -filelist "+ datafolder + trialname + "/tmp/" + trialname + "/list.txt -output " + str(NbestNum+1) +" -dnnconf " + Juliusfolder + "julius.dnnconf $*" ) #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS 
      print "Read dic: " + str(step-1) + "/WD.htkdic" , step , nbest

    startWordGraphData = False
    wordGraphData = []
    wordData = []
    index = 1 ###単語IDを1から始める
    line = p.readline()  #一行ごとに読む？
    #print line
    ncount = 0
    while line:
        if line.find( "sentence" + str(NbestNum+1) + ":" ) != -1:
            startWordGraphData = False

        if startWordGraphData==True:
            items = line#.split()  #空白で区切る
            wordData = []
            #wordData["range"] = items[1][1:-1].split("..")
            #wordData["index"] = str(index)
            #index += 1
            ##print items
            for n in range(1,nbest+1):
              #if ( 'pass1_best:') in items:
              #  name = items.replace('pass1_best:','')   #各name=valueをイコールで区切り格納
              #  #if name in ( "right" , "right_lscore" , "left" ):
              #  #    value = value.split(",")
              #  wordData = name.split(":")
              #  #print n,index

              #  #wordData[name] = value
              #  wordData = wordData.decode("sjis")
              #  ##print wordData
              #  
              #  #if (int(n) == int(index)): 
              #  #if (n ==  1):
              #  if ( (len(wordGraphData) == 0) ):
              #    wordGraphData.append(wordData)
              #  else:
              #    wordGraphData[-1] = wordData
              #  #else:
              #  #  wordGraphData[-1] = wordData
              if ( 'sentence' + str(n) + ":" ) in items : 
                name = items.replace('sentence','')   #各name=valueをイコールで区切り格納
                #if name in ( "right" , "right_lscore" , "left" ):
                #    value = value.split(",")
                index, wordData = name.split(":")
                #print n,index
                ncount += 1

                #wordData[name] = value
                wordData = wordData.decode("sjis")
                #####wordData = wordData.decode("utf8")
                #print n,wordData
                
                #if (int(n) == int(index)): 
                if ( (n == 1) and (len(wordGraphData) == 0) ):
                  wordGraphData.append(wordData)
                else:
                  wordGraphData.append(wordData)
                  #wordGraphData[-1] = wordData
                wordDataLast = wordData

        if line.find("Stat: adin_file: input speechfile:") != -1:
            startWordGraphData = True
        line = p.readline()
    p.close()
    
    #erorr処理：指定したN数よりも少ない認識結果のとき、最後の認識結果をコピーする
    while (ncount < NbestNum):
      wordGraphData.append(wordDataLast)
      ncount += 1
    
    #print wordGraphData
    return wordGraphData

# 認識したlatticeをopenFST形式で保存
def SaveLattice( wordGraphData , filename):
    f = codecs.open( filename , "w" , "sjis" )
    #print wordGraphData
    #wordGraphData = wordGraphData.decode("sjis")
    
    for wordData in wordGraphData:
        sentence = wordData#.decode("sjis")
        ##print sentence
        f.write("%s\n" % sentence)
    
    #f.write( "%d 0" % int(len(wordGraphData)+1) )
    f.close()

# テキスト形式をバイナリ形式へコンパイル
#def FSTCompile( txtfst , syms , outBaseName , filename ):
#    Makedir( "tmp" )
#    Makedir( "tmp/" + filename )
#    os.system( "fstcompile --isymbols=%s --osymbols=%s %s %s.fst" % ( syms , syms , txtfst , outBaseName ) )
#    os.system( "fstdraw  --isymbols=%s --osymbols=%s %s.fst > tmp/%s/fst.dot" % ( syms , syms , outBaseName , filename ) )
#
#    # sjisをutf8に変換して，日本語フォントを指定
#    #codecs.open( "tmp/" + filename + "/fst_utf.dot" , "w" , "utf-8" ).write( codecs.open( "tmp/" + filename + "/fst.dot" , "r" , "sjis" ).read().replace( 'label' , u'fontname="MS UI Gothic" label' ) )
#
#    # psとして出力
#    #os.system( "dot -Tps:cairo tmp/%s/fst_utf.dot > %s.ps" % (filename , outBaseName) )
#    # pdf convert
#    #os.system( "ps2pdf %s.ps %s.pdf" % (outBaseName, outBaseName) )


def Julius_lattice(step, filename, trialname):
    step = int(step)
    #Makedir( "data/" + filename + "/fst_gmm_" + str(iteration+1) )
    #Makedir( "data/" + filename + "/out_gmm_" + str(iteration+1) )
    #Makedir( "data/" + filename + "/out_gmm_" + str(iteration) )
    Makedir( filename + "/fst_gmm" )
    Makedir( filename + "/out_gmm" )

    # wavファイルを指定
    files = glob.glob(speech_folder)   #./../../../Julius/directory/CC3Th2/ (相対パス)
    #print files
    files.sort()
    
    #step分までのみ使用
    files2 = [files[i] for i in xrange(step)]

    wordDic = [] #set()
    num = 0

    # 1つづつ認識してFSTフォーマットで保存
    for f in files2:
        #Nbest認識結果をnごとに処理、保存
        #for nbest in range(1,N_best_number+1):
        #wordDic = []
        #num = 0
        ######file = open( "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/sentences" + str(nbest) + ".word", "w" )
        #n=10  #n-bestのnをどこまでとるか（n<=10）
        
        
        #txtfstfile = filename + "/fst_gmm/" + str(nbest) + "%03d.txt" % num
        txtfstfile = filename + "/fst_gmm/" + "%03d.txt" % num
        print "count...", f , num

        # Lattice認識&保存
        graph = RecogLattice( f , step ,filename, trialname ,NbestNum)
        ######
        SaveLattice( graph , txtfstfile) #wavfile , step , filename , trialname, nbest
        
        ######b = 1
        ######for line in open( txtfstfile , "r" ):
        ######  if (b <= N_best_number):
        ######    file.write(line)
        ######    ##print line
        ######    b = b+1
        

        # 単語辞書に追加
        for word in graph:
            for i in range(5):
              word = word.replace(" <s> ", "")
              word = word.replace("<sp> ", "")
              word = word.replace(" </s>", "")
              word = word.replace(" ", "") 
              word = word.replace("\n", "")           
            word2 = ""
            for w in range(len(word)): #空白をいれる処理
              if ((word[w] or word[w].encode('sjis') or word[w].decode('sjis')) == (u'ぁ' or u'ぃ' or u'ぅ' or u'ぇ' or u'ぉ' or u'ゃ' or u'ゅ' or u'ょ' or "ぁ".decode('sjis') or "ぃ".decode('sjis') or "ぅ".decode('sjis') or "ぇ".decode('sjis') or "ぉ".decode('sjis') or "ゃ".decode('sjis') or "ゅ".decode('sjis') or "ょ".decode('sjis'))):   ##################################ぁ　ぃ　ぅ　ぇ　ぉ　ゃ　ゅょ
                  #print word[w]
                  word2 = word2[:-1] + word[w] + " "
                #print u'ぁ',"ぁ".decode('sjis'),word[w]
              else:
                  word2 = word2 + word[w] + " "
            word2 = word2.replace("  ", " ")
            word2 = word2.replace(" ぁ".decode('sjis'), "ぁ".decode('sjis'))
            word2 = word2.replace(" ぃ".decode('sjis'), "ぃ".decode('sjis'))
            word2 = word2.replace(" ぅ".decode('sjis'), "ぅ".decode('sjis'))
            word2 = word2.replace(" ぇ".decode('sjis'), "ぇ".decode('sjis'))
            word2 = word2.replace(" ぉ".decode('sjis'), "ぉ".decode('sjis'))
            word2 = word2.replace(" ゃ".decode('sjis'), "ゃ".decode('sjis'))
            word2 = word2.replace(" ゅ".decode('sjis'), "ゅ".decode('sjis'))
            word2 = word2.replace(" ょ".decode('sjis'), "ょ".decode('sjis'))
            
            wordDic.append( word2 )
            #print wordDic
            print num,len(wordDic),word2

        num += 1
        
        #file.close()
      
    # 現在までのデータのすべてのN-best認識結果のリストを作成
    f = open( filename + "/fst_gmm/sentences" + str(NbestNum) + ".char" , "w")# , "sjis" )
    #wordDic = list(wordDic)
    #f.write( "<eps>	0\n" )  # latticelmでこの2つは必要らしい.decode("sjis")
    #f.write( "<phi>	1\n" )
    for i in range(len(wordDic)):
        f.write(wordDic[i].encode('sjis'))
        f.write('\n')
    f.close()
      
      

