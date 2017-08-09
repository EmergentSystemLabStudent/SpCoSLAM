# encoding: shift_jis
# Akira Taniguchi 2017/02/25-2017/06/28-2071/07/06
# ���s����΁A�����I�Ɏw��t�H���_���ɂ��鉹���t�@�C����ǂݍ��݁AJulius�Œʏ�̉����F���������ʂ�N-best�ŏo�͂��Ă����B
# ���ӓ_�F�w��t�H���_���������Ă��邩�m�F���邱�ƁB
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


# julius�ɓ��͂��邽�߂�wav�t�@�C���̃��X�g�t�@�C�����쐬
def MakeTmpWavListFile( wavfile , trialname):
    Makedir( datafolder + trialname + "/" + "tmp" )
    Makedir( datafolder + trialname + "/" + "tmp/" + trialname )
    fList = codecs.open( datafolder + trialname + "/" + "tmp/" + trialname + "/list.txt" , "w" , "sjis" )
    fList.write( wavfile )
    fList.close()


# Lattice�F��
def RecogLattice( wavfile , step , filename , trialname, nbest):
    MakeTmpWavListFile( wavfile , trialname )
    if (JuliusVer == "v4.4"):
      binfolder = "bin/linux/julius"
    else:
      binfolder = "bin/julius"
    if (step == 0 or step == 1):  #�ŏ��͓��{�ꉹ�߂݂̂̒P�ꎫ�����g�p(step==1���ŏ�)####�g�p����Ȃ��͂�
      p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + lmfolder + lang_init + " -demo -filelist "+ datafolder + trialname + "/tmp/" + trialname + "/list.txt -output " + str(NbestNum+1) ) #���ݒ�-n 5 # -gram type -n 5-charconv UTF-8 SJIS 
      print "Read dic:" ,lang_init , step
    else:  #�X�V�����P�ꎫ�����g�p
      #print Juliusfolder + "bin/julius -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + datafolder + trialname + "/" + str(step-1) + "/WD.htkdic -demo -filelist tmp/" + trialname + "/list.txt -confnet -lattice"
      p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + datafolder + trialname + "/" + str(step-1) + "/WD.htkdic -demo -filelist "+ datafolder + trialname + "/tmp/" + trialname + "/list.txt -output " + str(NbestNum+1) ) #���ݒ�-n 5 # -gram type -n 5-charconv UTF-8 SJIS 
      print "Read dic: " + str(step-1) + "/WD.htkdic" , step , nbest

    startWordGraphData = False
    wordGraphData = []
    wordData = []
    index = 1 ###�P��ID��1����n�߂�
    line = p.readline()  #��s���ƂɓǂށH
    #print line
    ncount = 0
    while line:
        if line.find( "sentence" + str(NbestNum+1) + ":" ) != -1:
            startWordGraphData = False

        if startWordGraphData==True:
            items = line#.split()  #�󔒂ŋ�؂�
            wordData = []
            #wordData["range"] = items[1][1:-1].split("..")
            #wordData["index"] = str(index)
            #index += 1
            ##print items
            for n in range(1,nbest+1):
              #if ( 'pass1_best:') in items:
              #  name = items.replace('pass1_best:','')   #�ename=value���C�R�[���ŋ�؂�i�[
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
                name = items.replace('sentence','')   #�ename=value���C�R�[���ŋ�؂�i�[
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
    
    #erorr�����F�w�肵��N���������Ȃ��F�����ʂ̂Ƃ��A�Ō�̔F�����ʂ��R�s�[����
    while (ncount < NbestNum):
      wordGraphData.append(wordDataLast)
      ncount += 1
    
    #print wordGraphData
    return wordGraphData

# �F������lattice��openFST�`���ŕۑ�
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

# �e�L�X�g�`�����o�C�i���`���փR���p�C��
#def FSTCompile( txtfst , syms , outBaseName , filename ):
#    Makedir( "tmp" )
#    Makedir( "tmp/" + filename )
#    os.system( "fstcompile --isymbols=%s --osymbols=%s %s %s.fst" % ( syms , syms , txtfst , outBaseName ) )
#    os.system( "fstdraw  --isymbols=%s --osymbols=%s %s.fst > tmp/%s/fst.dot" % ( syms , syms , outBaseName , filename ) )
#
#    # sjis��utf8�ɕϊ����āC���{��t�H���g���w��
#    #codecs.open( "tmp/" + filename + "/fst_utf.dot" , "w" , "utf-8" ).write( codecs.open( "tmp/" + filename + "/fst.dot" , "r" , "sjis" ).read().replace( 'label' , u'fontname="MS UI Gothic" label' ) )
#
#    # ps�Ƃ��ďo��
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

    # wav�t�@�C�����w��
    files = glob.glob(speech_folder)   #./../../../Julius/directory/CC3Th2/ (���΃p�X)
    #print files
    files.sort()
    
    #step���܂ł̂ݎg�p
    files2 = [files[i] for i in xrange(step)]

    wordDic = [] #set()
    num = 0

    # 1�ÂF������FST�t�H�[�}�b�g�ŕۑ�
    for f in files2:
        #Nbest�F�����ʂ�n���Ƃɏ����A�ۑ�
        #for nbest in range(1,N_best_number+1):
        #wordDic = []
        #num = 0
        ######file = open( "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/sentences" + str(nbest) + ".word", "w" )
        #n=10  #n-best��n���ǂ��܂łƂ邩�in<=10�j
        
        
        #txtfstfile = filename + "/fst_gmm/" + str(nbest) + "%03d.txt" % num
        txtfstfile = filename + "/fst_gmm/" + "%03d.txt" % num
        print "count...", f , num

        # Lattice�F��&�ۑ�
        graph = RecogLattice( f , step ,filename, trialname ,NbestNum)
        ######
        SaveLattice( graph , txtfstfile) #wavfile , step , filename , trialname, nbest
        
        ######b = 1
        ######for line in open( txtfstfile , "r" ):
        ######  if (b <= N_best_number):
        ######    file.write(line)
        ######    ##print line
        ######    b = b+1
        

        # �P�ꎫ���ɒǉ�
        for word in graph:
            for i in range(5):
              word = word.replace(" <s> ", "")
              word = word.replace("<sp> ", "")
              word = word.replace(" </s>", "")
              word = word.replace(" ", "") 
              word = word.replace("\n", "")           
            word2 = ""
            for w in range(len(word)): #�󔒂�����鏈��
              if ((word[w] or word[w].encode('sjis') or word[w].decode('sjis')) == (u'��' or u'��' or u'��' or u'��' or u'��' or u'��' or u'��' or u'��' or "��".decode('sjis') or "��".decode('sjis') or "��".decode('sjis') or "��".decode('sjis') or "��".decode('sjis') or "��".decode('sjis') or "��".decode('sjis') or "��".decode('sjis'))):   ##################################���@���@���@���@���@��@���
                  #print word[w]
                  word2 = word2[:-1] + word[w] + " "
                #print u'��',"��".decode('sjis'),word[w]
              else:
                  word2 = word2 + word[w] + " "
            word2 = word2.replace("  ", " ")
            word2 = word2.replace(" ��".decode('sjis'), "��".decode('sjis'))
            word2 = word2.replace(" ��".decode('sjis'), "��".decode('sjis'))
            word2 = word2.replace(" ��".decode('sjis'), "��".decode('sjis'))
            word2 = word2.replace(" ��".decode('sjis'), "��".decode('sjis'))
            word2 = word2.replace(" ��".decode('sjis'), "��".decode('sjis'))
            word2 = word2.replace(" ��".decode('sjis'), "��".decode('sjis'))
            word2 = word2.replace(" ��".decode('sjis'), "��".decode('sjis'))
            word2 = word2.replace(" ��".decode('sjis'), "��".decode('sjis'))
            
            wordDic.append( word2 )
            #print wordDic
            print num,len(wordDic),word2

        num += 1
        
        #file.close()
      
    # ���݂܂ł̃f�[�^�̂��ׂĂ�N-best�F�����ʂ̃��X�g���쐬
    f = open( filename + "/fst_gmm/sentences" + str(NbestNum) + ".char" , "w")# , "sjis" )
    #wordDic = list(wordDic)
    #f.write( "<eps>	0\n" )  # latticelm�ł���2�͕K�v�炵��.decode("sjis")
    #f.write( "<phi>	1\n" )
    for i in range(len(wordDic)):
        f.write(wordDic[i].encode('sjis'))
        f.write('\n')
    f.close()
      
      

