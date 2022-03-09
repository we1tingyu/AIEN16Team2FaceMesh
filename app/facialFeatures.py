#  facial research: https://www.researchgate.net/figure/Linear-measures-11-to-17-11-Upper-facial-width-Zid-and-Zie-12-Lower-facial-width_fig4_262762692


# 臉部特徵座標[[起點id,終點id]]


# 上下左右
# EDGE = {'top':10, 'bottom':152, 'left':234, 'right':454}
EDGE = [10, 152, 234, 454]

# 眉尾
END_OF_EYEBROW = [70, 300]
# 眼頭
HEAD_OF_EYE = [243, 463]
# 眼尾
END_OF_EYE = [130, 359]
# 鼻翼
ALAE_OF_NOSE = [129, 358]
# 唇角
LIP_CORNER = [61, 291]

# 三庭
THREE_COURT = [EDGE[0], 9, 94, EDGE[1]]
# 五眼
FIVE_EYE = [EDGE[2], END_OF_EYE[0], HEAD_OF_EYE[0], HEAD_OF_EYE[1], END_OF_EYE[1], EDGE[3]]

# 美人角
BEAUTY_CORNER_LEFT = [[END_OF_EYE[0], LIP_CORNER[0]]]
BEAUTY_CORNER_RIGHT = [[END_OF_EYE[1], LIP_CORNER[1]]]
BEAUTY_CORNER = [BEAUTY_CORNER_LEFT, BEAUTY_CORNER_RIGHT]

# 眉尾和眼尾
EYEBROW_AND_EYE_LEFT = [[END_OF_EYEBROW[0], END_OF_EYE[0]]]
EYEBROW_AND_EYE_RIGHT = [[END_OF_EYEBROW[1], END_OF_EYE[1]]]
EYEBROW_AND_EYE = [EYEBROW_AND_EYE_LEFT, EYEBROW_AND_EYE_RIGHT]

# 眼尾和鼻翼
EYE_AND_NOSE_LEFT = [[END_OF_EYE[0], ALAE_OF_NOSE[0]]]
EYE_AND_NOSE_RIGHT = [[END_OF_EYE[1], ALAE_OF_NOSE[1]]]
EYE_AND_NOSE = [EYE_AND_NOSE_LEFT, EYE_AND_NOSE_RIGHT]

# 左眉
RIGHT_EYEBROW = [[46,53],[53,52],[52,65],[65,55],[55,107],[107,66],[66,105],[105,63],[63,70],[70,46]]
# 右眉
LEFT_EYEBROW = [[276, 283], [283, 282], [282, 295],[295, 285],[285,336],[336,296],[296,334],[334,293],[293,300],[300,276]]
# 左眼
RIGHT_EYE = [[33, 7], [7, 163], [163, 144], [144, 145],[145, 153], [153, 154], [154, 155], [155, 133],[33, 246], [246, 161], [161, 160], [160, 159],[159, 158], [158, 157], [157, 173], [173, 133]]
# 左眼
LEFT_EYE = [[263, 249], [249, 390], [390, 373], [373, 374],[374, 380], [380, 381], [381, 382], [382, 362],[263, 466], [466, 388], [388, 387], [387, 386],[386, 385], [385, 384], [384, 398], [398, 362]]
# 鼻長
NOSE_LENGTH = [[1,168]]
# 鼻寬
NOSE_WIDTH = [ALAE_OF_NOSE]
# 前額
FOREHEAD = [[9,10]]
# 人中
PHILTRUM = [[2,0]]
# 嘴
MOUTH =[[61, 146], [146, 91], [91, 181], [181, 84], [84, 17],[17, 314], [314, 405], [405, 321], [321, 375],[375, 291], [61, 185], [185, 40], [40, 39], [39, 37],
                           [37, 0], [0, 267],[267, 269], [269, 270], [270, 409], [409, 291],[78, 95], [95, 88], [88, 178], [178, 87], [87, 14],[14, 317], [317, 402], [402, 318], [318, 324],
                           [324, 308], [78, 191], [191, 80], [80, 81], [81, 82],[82, 13], [13, 312], [312, 311], [311, 310],[310, 415], [415, 308]]
# 五官
READ_FACE = [RIGHT_EYEBROW,LEFT_EYEBROW,RIGHT_EYE,LEFT_EYE,NOSE_LENGTH,NOSE_WIDTH,FOREHEAD,PHILTRUM,MOUTH]
# 臉部外框
FACE_OVAL = [[[10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10]]]

