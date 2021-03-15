import matplotlib
import matplotlib.pyplot as plt
import numpy as np

'''
# topic domain
labels = ['50k', '100k', '200k', '300k', '400k']
x_coordinates = [50000, 100000, 200000, 300000, 400000]


cnn = [0.965, 0.9688, 0.9706, 0.968, 0.9656]
lstm = [0.9994, 0.998, 0.99915, 0.9991, 0.9996]
cnnlstm = [0.9678, 0.9696, 0.9738, 0.9684, 0.9724]

x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()
plt.ylim(0, 1.4)

ax.bar(x - width/2, cnn, width, label='CNN')
ax.bar(x + width/2, lstm, width, label='LSTM')
ax.bar(x + width, cnnlstm, width, label='CLSTM')

ax.set_ylabel('Scores')
ax.set_xlabel('Data batches')
ax.set_title('Event')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()



# Polarity
cnn = [0.921, 0.9306, 0.9248, 0.9264, 0.9272]
lstm = [0.9926, 0.9936, 0.9929, 0.9916, 0.9901]
cnnlstm = [0.9162, 0.9212, 0.9106, 0.9174, 0.912]

fig, ax = plt.subplots()
cnn = ax.bar(x - width/2, cnn, width, label='CNN')
lstm = ax.bar(x + width/2, lstm, width, label='LSTM')
cnnlstm = ax.bar(x + width, cnnlstm, width, label='CLSTM')
plt.ylim(0, 1.4)
ax.set_ylabel('Scores')
ax.set_xlabel('Data batches')
ax.set_title('Sentiment/Polarity')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
'''
########### LOSS ########

# importing package 
import matplotlib.pyplot as plt 
import numpy as np 
  
# create data 
x = np.arange(20) # [1,2,3,4,5] 
# y = [3,3,3,3,3] 

#50000
# cnn = [0.8939094543457031, 0.6821804642677307, 0.5978307723999023, 0.5444413423538208, 0.5066081881523132, 0.4756433069705963, 0.4552914798259735, 0.43036210536956787, 0.414082407951355, 0.3991740942001343, 0.38578924536705017, 0.3757725656032562, 0.367031067609787, 0.353927880525589, 0.34723523259162903, 0.3370286822319031, 0.33036231994628906, 0.3227541148662567, 0.3183358907699585, 0.3128716051578522]
# lstm = [0.7674027681350708, 0.4653863310813904, 0.3256138563156128, 0.2498251050710678, 0.19639486074447632, 0.15782299637794495, 0.13065868616104126, 0.10917860269546509, 0.09299822151660919, 0.08035875856876373, 0.0696183517575264, 0.06279131770133972, 0.0563090555369854, 0.05039919167757034, 0.04980470985174179, 0.04490409046411514, 0.041217777878046036, 0.04004927724599838, 0.03666092827916145, 0.03389640152454376]
# clstm = [0.8692082762718201, 0.6529467701911926, 0.5683695673942566, 0.5171769857406616, 0.48436054587364197, 0.45724183320999146, 0.4441207945346832, 0.42388832569122314, 0.40806102752685547, 0.39767786860466003, 0.38520047068595886, 0.3784162402153015, 0.3665250539779663, 0.3619235157966614, 0.35332784056663513, 0.3470991849899292, 0.33924582600593567, 0.33651232719421387, 0.3268541991710663, 0.32245081663131714]  

#100k
# cnn = [0.902153491973877, 0.6865175366401672, 0.6029682755470276, 0.5495494604110718, 0.5133033990859985, 0.4823238253593445, 0.4586125314235687, 0.4375040829181671, 0.4157205820083618, 0.40341177582740784, 0.3885710835456848, 0.37783172726631165, 0.3656541109085083, 0.35471662878990173, 0.34855952858924866, 0.33858296275138855, 0.3322577476501465, 0.321365624666214, 0.3179819583892822, 0.3127528131008148]
# lstm = [0.6207547187805176, 0.3270217776298523, 0.23307563364505768, 0.18390846252441406, 0.15185533463954926, 0.1286161243915558, 0.11028128117322922, 0.09644096344709396, 0.08528509736061096, 0.07590912282466888, 0.0691261887550354, 0.06175893917679787, 0.058322370052337646, 0.054230887442827225, 0.05061908811330795, 0.04743022099137306, 0.04309428855776787, 0.0429108552634716, 0.04011480510234833, 0.039150021970272064]
# clstm = [0.876170814037323, 0.6555135250091553, 0.5664563179016113, 0.5176525712013245, 0.48119282722473145, 0.4578169584274292, 0.4344686269760132, 0.4140975773334503, 0.4024333357810974, 0.3892655074596405, 0.3780389428138733, 0.36688393354415894, 0.3626285195350647, 0.3530533015727997, 0.3459225594997406, 0.33927398920059204, 0.33212995529174805, 0.3279270529747009, 0.3200490176677704, 0.31495925784111023]


#200k
# cnn = [0.8947420120239258, 0.6868395209312439, 0.6058568358421326, 0.550259530544281, 0.510299026966095, 0.4791121482849121, 0.4518054723739624, 0.42950108647346497, 0.41244813799858093, 0.3967382311820984, 0.38340917229652405, 0.37058624625205994, 0.3593815863132477, 0.3524331748485565, 0.3396373391151428, 0.3333158791065216, 0.3268811106681824, 0.3186151683330536, 0.31487223505973816, 0.3045518100261688]
# lstm = [0.4860638976097107, 0.2246488481760025, 0.16390199959278107, 0.13390839099884033, 0.11493887007236481, 0.10066273808479309, 0.09047038108110428, 0.08122526854276657, 0.07432635873556137, 0.06788628548383713, 0.06301948428153992, 0.058840516954660416, 0.05512420833110809, 0.05221245065331459, 0.04934396222233772, 0.04721638932824135, 0.04471217840909958, 0.04318220168352127, 0.04115169122815132, 0.03942948579788208]
# clstm = [0.8773103356361389, 0.6600216627120972, 0.5732192993164062, 0.5176407694816589, 0.48222169280052185, 0.4542882442474365, 0.4353301227092743, 0.416548490524292, 0.4001024663448334, 0.3855557143688202, 0.37591496109962463, 0.3616866171360016, 0.35679879784584045, 0.3525438606739044, 0.3404248356819153, 0.3337724208831787, 0.32936611771583557, 0.3212568461894989, 0.31696152687072754, 0.312648206949234]

#300k
# cnn = [0.9017605781555176, 0.6916912794113159, 0.6084070205688477, 0.5549095273017883, 0.5159263610839844, 0.481336772441864, 0.45936262607574463, 0.4349653720855713, 0.41494959592819214, 0.39887070655822754, 0.38458722829818726, 0.3729826807975769, 0.36479318141937256, 0.354147344827652, 0.34442609548568726, 0.3375123143196106, 0.3250882923603058, 0.3207279443740845, 0.31345057487487793, 0.30986958742141724]
# lstm = [0.42429786920547485, 0.1961352378129959, 0.14920589327812195, 0.12552565336227417, 0.10964184254407883, 0.09740714728832245, 0.08887029439210892, 0.081112802028656, 0.07464848458766937, 0.06896812468767166, 0.06470534205436707, 0.06026419252157211, 0.057614970952272415, 0.05521319806575775, 0.05223383009433746, 0.05056281015276909, 0.04851881042122841, 0.04676753655076027, 0.04384765028953552, 0.04292893037199974]
# clstm = [0.8696476817131042, 0.6481075286865234, 0.564065158367157, 0.5123823285102844, 0.47684499621391296, 0.45236557722091675, 0.4307956099510193, 0.41835999488830566, 0.4032374620437622, 0.39227911829948425, 0.3800071179866791, 0.36794236302375793, 0.36190956830978394, 0.3518788814544678, 0.3476478159427643, 0.3379669785499573, 0.33303946256637573, 0.3278372883796692, 0.3226105570793152, 0.3209969401359558]

#400k
cnn = [0.8958412408828735, 0.6862093210220337, 0.6007915139198303, 0.5483180284500122, 0.5080769062042236, 0.4783443510532379, 0.45344722270965576, 0.43267032504081726, 0.412772536277771, 0.4009860157966614, 0.3872625529766083, 0.3724467158317566, 0.36338895559310913, 0.3538009524345398, 0.3414287567138672, 0.3357713520526886, 0.3308451771736145, 0.3232421576976776, 0.3184252083301544, 0.30928781628608704] 
lstm = [0.3746325969696045, 0.1711798906326294, 0.13173404335975647, 0.11213508248329163, 0.09831622242927551, 0.08891016244888306, 0.0814308226108551, 0.07516803592443466, 0.06989600509405136, 0.06602735817432404, 0.06264511495828629, 0.059120822697877884, 0.05678649991750717, 0.054121147841215134, 0.051967453211545944, 0.04981311410665512, 0.04787503927946091, 0.046481598168611526, 0.04520054906606674, 0.04372655600309372]
clstm = [0.8592146039009094, 0.642196774482727, 0.5595354437828064, 0.5091079473495483, 0.4764961004257202, 0.4483361542224884, 0.42733052372932434, 0.40921658277511597, 0.39316806197166443, 0.37948793172836304, 0.36923927068710327, 0.35980433225631714, 0.35179027915000916, 0.34339284896850586, 0.3327583074569702, 0.3246872127056122, 0.32333555817604065, 0.31563475728034973, 0.3119807839393616, 0.3026787042617798]





# plot lines 
# plt.plot(x, y, label = "line 1") 
# plt.plot(y, x, label = "line 2") 
plt.plot(x, cnn, label = "CNN", color='r') 
plt.plot(x, lstm, label = "LSTM", color='b') 
plt.plot(x, clstm, label = "CLSTM", color='y') 
x_ticks = np.arange(0, 20, 5)
plt.xticks(x_ticks)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('400000 Dataset')
plt.legend() 
plt.show()
