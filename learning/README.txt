�w�K�p�v���O����

�y�t�H���_�z
/lamg_m/�FJulus�p�̒P�ꎫ���t�@�C���i���{�ꉹ�ߎ����Ȃǁj�������Ă���t�H���_


�y�t�@�C���z
CNN_place.py�FCNN�̓����ʂ����̂܂ܕۑ�����.(PlaceCNN��)
Julius1best_gmm.py�F���PRR�]���p�Ɏg�p�BJulius�ŉ����F�����An-best�����F�����ʂ𓾂�
JuliusLattice_gmm.py�F�w�K�v���O�����ƘA�g���āAJulius�ŉ����F����WFST�����F�����ʂ𓾂�
README.txt�F���̃t�@�C��
SpCoSLAM.sh�F�I�����C���w�K���s�p��{�t�@�C��
__init__.py�F�n�C�p�[�p�����[�^�Ȃǂ̏����l��t�@�C���p�X�w��p�̃t�@�C��
autovisualization.py�F�����Ŋw�K���ʂ𒀎��`�悷�邽�߂̃v���O����(�ۑ��͕ʂł��Ȃ���΂Ȃ�Ȃ�)
collectmapclocktime.py�F����쐬�̂��߂ɁA�n�}�̎������ЂƂ̃t�@�C���ɂ܂Ƃ߂Đ������邽�߂̂��́B
gmapping.sh�FFastSLAM���s�p�̃V�F���X�N���v�g
learnSpCoSLAM3.2.py�FSpCoSLAM online learning program (���ʂȃR�����g�A�E�g�R�[�h���Ȃ����o�[�W����+bugfix)�A���ASpCoA�̃I�����C���ŁiSpCoSLAM���猾�ꃂ�f���X�V�Ɖ摜������������o�[�W�����j
map_saver.py�F�n�}�𒀎��I�ɕۑ����Ă����v���O�����irospy�g�p�j
new_place_draw.py�F�w�K�����ꏊ�̈�̃T���v����rviz��ɉ�������v���O����(�Ε��i�T���v���v���b�g�j����Ӂi�K�E�X�T�`�j����)
new_place_draw_online.py�F�I�����C�������p
new_position_draw_online.py�F���{�b�g�̎��Ȉʒu�`��p
run_SpCoSLAM.py�FSpCoSLAM�����s���邽�߂̎q�v���O����
run_gmapping.sh�Fgmapping�����s���邽�߂̎q�v���O����
run_mapviewer.py�Fmap_server�R�}���h�����s���鏬�v���O�����i���g�p�H�j
run_mapviewer.sh�Frun_mapviewer.py�����s���邽�߂̃V�F���X�N���v�g�i���g�p�H�j
run_rosbag.py�Frosbag�����s���邽�߂̎q�v���O����
run_roscore.sh�Froscore�����s���邽�߂̎q�v���O����
saveSpCoMAP.rviz�Frviz�t�@�C��
saveSpCoMAP_online.rviz�Frviz�t�@�C���i�I�����C���`��p�j



�y���s�����z
�E�w�K�f�[�^�Z�b�g�̃p�X�w��A�g�s�b�N�������킹��Ȃǁi__init__.py�Arun_gmapping.sh�j
�E�w�K�f�[�^�Z�b�g�̎��������A����������ۑ������t�@�C�����쐬�i�j
�E�����f�[�^�t�@�C����p�ӂ��Ă����B__init__.py�Ńt�@�C���p�X�w��B
�E�w�K�v���O�������s�̑O�ɁACNN_place.py�����s�B�摜�����̃t�@�C���t�H���_���쐬���Ă����B
�E�p�[�e�B�N�����̎w��́A__init__.py��run_gmapping.sh�̗����ύX����K�v������B

�y���s���@�z
cd ~/Dropbox/SpCoSLAM/learning
./SpCoSLAM.sh

�y���ӎ����z
�Erun_rosbag.py�ł��܂�gflag�֌W�̃G���[���ł邱�Ƃ����邪�A�t�@�C���ǂݍ��ݎ��s�������Bcallback�֐��Ȃ̂łق��Ă����΍ēǂݍ��݂��Ă���āA�����̂Ŗ��Ȃ��B
�E��X�y�b�NPC�ł́Agmapping�̏������ǂ������ɒn�}�����܂��ł��Ȃ����Ƃ�����B

-----
[rviz��Ɉʒu���z��`�悷����@]
Google�h���C�u�̃t�@�C���Q�ƁB
roscore
rviz -d ./Dropbox/SpCoSLAM/learning/saveSpCoMAP_online.rviz 
python ./autovisualization.py p30a20g10sfix008

�ʎw��̏ꍇ
rosrun map_server map_server ./p30a20g10sfix008/map/map361.yaml
python ./new_place_draw.py p30a20g10sfix008 50 23 

-------------------------------------------------
�X�V����
2017/02/12 Akira Taniguchi
2017/03/12 Akira Taniguchi
2018/01/12 Akira Taniguchi