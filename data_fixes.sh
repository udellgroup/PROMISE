#!/bin/bash
# Make fixes to HIGGS, SUSY, webspam_wc_normalized_unigram.svm
# Takes in the path to the data directory as an argument
sed -i '3124994s/.*/0 1:5.291303396224975586e-01 2:-1.670747995376586914e+00 3:-2.836283147335052490e-01 4:8.541975617408752441e-01 5:-1.992583572864532471e-01 6:-1.642527222633361816e+00 7:3.792838156223297119e-01 8:5.899490714073181152e-01 9:5.693517923355102539e-01 10:-3.471146151423454285e-02 11:7.617459297180175781e-01 12:5.705336928367614746e-01 13:6.646450757980346680e-01 14:8.746069669723510742e-01 15:7.685849070549011230e-01 16:6.556583046913146973e-01 17:1.039212822914123535e+00 18:2.716330066323280334e-02/' $1/SUSY
sed -i '4843758s/.*/0 1:6.499441266059875488e-01 2:-1.544686794281005859e+00 3:3.505432307720184326e-01 4:8.460544943809509277e-01 5:9.073476791381835938e-01 6:-9.537825584411621094e-01 7:4.845112860202789307e-01 8:1.441670298576354980e+00 9:7.273095250129699707e-01 10:-1.491162627935409546e-01 11:1.369668245315551758e+00 12:6.985642910003662109e-01 13:4.525914490222930908e-01 14:1.135811686515808105e+00 15:1.378148674964904785e+00 16:8.608778715133666992e-01 17:1.057872533798217773e+00 18:1.506430003792047501e-02/' $1/SUSY 
sed -i '1031260s/.*/0 1:3.088827371597290039e+00 2:-1.120118141174316406e+00 3:4.193368554115295410e-01 4:4.879521131515502930e-01 5:8.139222264289855957e-01 6:9.932970404624938965e-01 7:-4.991013407707214355e-01 8:-1.028310775756835938e+00 9:2.173076152801513672e+00 10:1.140811562538146973e+00 11:-7.862099260091781616e-02 12:-1.449421405792236328e+00 14:1.388637900352478027e+00 15:-1.482622265815734863e+00 16:6.300848126411437988e-01 18:9.405252337455749512e-01 19:-1.322983980178833008e+00 20:-1.142275929450988770e+00 22:1.176077485084533691e+00 23:1.039700627326965332e+00 24:9.888745546340942383e-01 25:1.578675627708435059e+00 26:9.644216895103454590e-01 27:1.278195381164550781e+00 28:1.162336826324462891e+00/' $1/HIGGS
sed -i '2062491s/.*/0 1:1.492256164550781250e+00 2:-2.630257308483123779e-01 3:-1.528886675834655762e+00 4:2.525255084037780762e-01 5:4.774572849273681641e-01 6:8.678867816925048828e-01 7:1.086256861686706543e+00 8:-1.643667340278625488e+00 9:1.086538076400756836e+00 10:6.708137989044189453e-01 11:1.406609296798706055e+00 12:-4.229834973812103271e-01 14:5.335376262664794922e-01 15:1.039412394165992737e-01 16:1.061114192008972168e+00 17:2.548224449157714844e+00 18:1.196832895278930664e+00 19:-6.292243301868438721e-02 20:-2.444254159927368164e-01 22:9.958372712135314941e-01 23:8.298872113227844238e-01 24:9.825775623321533203e-01 25:1.069854497909545898e+00 26:6.988831758499145508e-01 27:7.916187644004821777e-01 28:8.316564559936523438e-01/' $1/HIGGS
sed -i '3093765s/.*/1 1:1.324985861778259277e+00 2:1.163346972316503525e-02 3:1.544040799140930176e+00 4:1.181132346391677856e-01 5:-7.735881209373474121e-01 6:9.725022315979003906e-01 7:2.841705381870269775e-01 8:-1.934186816215515137e-01 10:6.666590571403503418e-01 11:-8.168649673461914062e-01 12:1.248266577720642090e+00 13:2.214872121810913086e+00 14:3.735466003417968750e-01 15:8.112044930458068848e-01 16:-5.432829260826110840e-01 18:6.608504056930541992e-01 19:9.547875523567199707e-01 20:1.691210508346557617e+00 21:3.101961374282836914e+00 22:4.727435111999511719e-01 23:1.103761434555053711e+00 24:9.817840456962585449e-01 25:6.529843211174011230e-01 26:7.358893156051635742e-01 27:7.497456669807434082e-01 28:7.289131283760070801e-01/' $1/HIGGS
sed -i '1375007s/.*/1 1:7.808997631072998047e-01 2:1.084947109222412109e+00 3:-1.184317708015441895e+00 4:1.604992389678955078e+00 5:-3.663000762462615967e-01 6:1.601935386657714844e+00 7:1.383325815200805664e+00 8:-9.723187685012817383e-01 9:2.173076152801513672e+00 10:1.802938103675842285e+00 11:-4.487143754959106445e-01 12:1.440792441368103027e+00 13:2.214872121810913086e+00 14:3.651343345642089844e+00 15:7.147181630134582520e-01 16:6.933246254920959473e-01 18:1.110910177230834961e+00 19:1.486146748065948486e-01 20:1.260153297334909439e-02 22:7.397826910018920898e-01 23:1.134726285934448242e+00 24:1.043434262275695801e+00 25:1.854036092758178711e+00 26:3.373844385147094727e+00 27:1.719359517097473145e+00 28:1.287033796310424805e+00/' $1/HIGGS
sed -i '233355s/.*/-1 10:0.270007 11:0.0703629 14:0.0699391 33:0.234402 34:0.00169549 35:0.156833 36:0.000847746 38:0.0131401 39:0.00211937 40:0.128857 41:0.000847746 42:0.000847746 45:0.0148356 46:0.00593422 47:0.0457783 48:0.276365 49:0.0309427 50:0.0118684 51:0.0046626 52:0.0186504 53:0.00169549 54:0.00423873 55:0.00169549 56:0.00254324 57:0.00127162 58:0.000847746 59:0.00381486 60:0.000847746 61:0.211513 62:0.144541 63:0.211513 64:0.000847746 65:0.000423873 66:0.00551035 67:0.0063581 68:0.0228891 69:0.0139878 70:0.00974908 71:0.00551035 72:0.00720584 73:0.00847746 74:0.00678197 75:0.00254324 76:0.000847746 77:0.00508648 78:0.00381486 79:0.0046626 80:0.00381486 81:0.0127162 83:0.00678197 84:0.010173 85:0.0152594 86:0.00127162 87:0.00551035 88:0.00678197 90:0.000423873 96:0.0220414 98:0.211937 99:0.0517125 100:0.0915566 101:0.194982 102:0.347576 103:0.0864701 104:0.119532 105:0.154714 106:0.270855 107:0.00169549 108:0.0474738 109:0.183113 110:0.0974908 111:0.184385 112:0.123347 113:0.0572229 114:0.000847746 115:0.158105 116:0.149627 117:0.372161 118:0.0682436 119:0.0618855 120:0.0322144 121:0.0046626 122:0.0309427 123:0.000423873/' $1/webspam_wc_normalized_unigram.svm
sed -i '306260s/.*/-1 10:0.153293 11:0.0818612 14:0.0707411 33:0.775702 34:0.00709924 35:0.193878 36:6.28252e-05 37:0.00012565 38:0.00395799 39:0.00282713 40:0.0152665 41:0.00634534 42:0.00628252 44:0.00628252 45:0.00370669 46:0.0307215 47:0.0216747 48:0.108311 49:0.0476843 50:0.0292137 51:0.00703642 52:0.010429 53:0.00282713 54:0.00910965 55:0.00452341 56:0.00238736 57:0.00458624 58:0.00301561 59:0.00389516 60:0.0102405 61:0.125588 62:0.109002 63:0.125588 64:0.00213606 65:6.28252e-05 66:0.0266379 67:0.00245018 68:0.00917248 69:0.0106175 70:0.0390144 71:0.00251301 72:0.00565427 73:0.0119996 74:0.026952 75:0.00276431 76:0.000314126 77:0.0160832 78:0.0111201 79:0.0180308 80:0.010052 81:0.019036 83:0.034868 84:0.0135074 85:0.0272033 86:0.00521449 87:0.00138215 88:0.00678512 89:0.000314126 90:0.000816727 91:0.000628252 96:0.0382605 98:0.171576 99:0.0511397 100:0.104604 101:0.134006 102:0.22083 103:0.0223029 104:0.0634534 105:0.0540925 106:0.150152 107:0.00665947 108:0.00376951 109:0.1232 110:0.0487523 111:0.0834947 112:0.0871385 113:0.0563542 114:0.00157063 115:0.134195 116:0.118865 117:0.216433 118:0.020858 119:0.0153922 120:0.0437263 121:0.00596839 122:0.00716207 123:0.00138215 125:0.000251301/' $1/webspam_wc_normalized_unigram.svm