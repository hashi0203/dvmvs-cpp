#pragma once

constexpr int tbbit = 10;
constexpr int actbit = 20;
constexpr qaint celu_table[1024] = {0, -3063, -6108, -9135, -12145, -15137, -18112, -21069, -24009, -26932, -29837, -32726, -35598, -38453, -41291, -44113, -46918, -49707, -52480, -55236, -57977, -60701, -63409, -66102, -68779, -71440, -74086, -76716, -79331, -81930, -84514, -87084, -89638, -92177, -94702, -97212, -99707, -102187, -104653, -107105, -109542, -111965, -114374, -116769, -119150, -121517, -123870, -126209, -128535, -130847, -133145, -135430, -137702, -139961, -142206, -144438, -146658, -148864, -151057, -153238, -155405, -157560, -159703, -161833, -163950, -166056, -168149, -170229, -172298, -174354, -176398, -178431, -180452, -182460, -184457, -186443, -188416, -190379, -192330, -194269, -196197, -198114, -200019, -201914, -203797, -205670, -207531, -209382, -211221, -213050, -214869, -216676, -218474, -220260, -222036, -223802, -225558, -227303, -229038, -230763, -232478, -234183, -235877, -237562, -239238, -240903, -242558, -244204, -245841, -247468, -249085, -250693, -252291, -253880, -255460, -257030, -258592, -260144, -261687, -263221, -264747, -266263, -267770, -269269, -270759, -272240, -273713, -275177, -276632, -278079, -279517, -280947, -282369, -283782, -285187, -286584, -287973, -289353, -290726, -292091, -293447, -294796, -296136, -297469, -298795, -300112, -301422, -302724, -304018, -305305, -306584, -307856, -309121, -310378, -311627, -312870, -314105, -315333, -316554, -317767, -318974, -320173, -321366, -322551, -323730, -324902, -326067, -327225, -328376, -329520, -330658, -331790, -332914, -334032, -335144, -336249, -337347, -338440, -339525, -340605, -341678, -342745, -343805, -344860, -345908, -346950, -347986, -349016, -350040, -351058, -352070, -353076, -354077, -355071, -356060, -357042, -358020, -358991, -359957, -360917, -361871, -362820, -363763, -364701, -365634, -366560, -367482, -368398, -369309, -370214, -371114, -372009, -372899, -373783, -374663, -375537, -376406, -377270, -378129, -378983, -379831, -380675, -381514, -382349, -383178, -384002, -384822, -385637, -386447, -387252, -388053, -388848, -389640, -390426, -391208, -391986, -392759, -393527, -394291, -395051, -395806, -396556, -397303, -398044, -398782, -399515, -400244, -400969, -401689, -402406, -403118, -403826, -404529, -405229, -405925, -406616, -407304, -407987, -408666, -409342, -410013, -410681, -411345, -412005, -412661, -413313, -413961, -414606, -415246, -415884, -416517, -417146, -417772, -418395, -419013, -419628, -420240, -420848, -421452, -422053, -422650, -423244, -423834, -424421, -425005, -425585, -426161, -426735, -427304, -427871, -428434, -428994, -429551, -430105, -430655, -431202, -431746, -432286, -432824, -433358, -433889, -434418, -434943, -435465, -435983, -436499, -437012, -437522, -438029, -438533, -439034, -439532, -440027, -440520, -441009, -441495, -441979, -442460, -442938, -443413, -443886, -444356, -444823, -445287, -445748, -446207, -446663, -447117, -447568, -448016, -448462, -448905, -449345, -449783, -450218, -450651, -451081, -451509, -451934, -452357, -452777, -453195, -453610, -454023, -454433, -454841, -455247, -455651, -456052, -456450, -456847, -457241, -457632, -458022, -458409, -458794, -459176, -459557, -459935, -460311, -460685, -461056, -461426, -461793, -462158, -462521, -462882, -463241, -463597, -463952, -464304, -464655, -465003, -465350, -465694, -466036, -466376, -466715, -467051, -467386, -467718, -468049, -468377, -468704, -469028, -469351, -469672, -469991, -470309, -470624, -470937, -471249, -471559, -471867, -472173, -472478, -472780, -473081, -473381, -473678, -473974, -474268, -474560, -474850, -475139, -475426, -475712, -475996, -476278, -476558, -476837, -477114, -477390, -477664, -477936, -478207, -478476, -478744, -479010, -479274, -479537, -479799, -480059, -480317, -480574, -480829, -481083, -481336, -481587, -481836, -482084, -482331, -482576, -482820, -483062, -483303, -483542, -483780, -484017, -484252, -484486, -484719, -484950, -485180, -485408, -485635, -485861, -486086, -486309, -486531, -486751, -486970, -487188, -487405, -487621, -487835, -488048, -488260, -488470, -488679, -488887, -489094, -489300, -489504, -489707, -489909, -490110, -490310, -490509, -490706, -490902, -491097, -491291, -491484, -491675, -491866, -492055, -492244, -492431, -492617, -492802, -492986, -493169, -493351, -493531, -493711, -493890, -494067, -494244, -494419, -494594, -494767, -494940, -495111, -495282, -495451, -495620, -495787, -495954, -496119, -496284, -496447, -496610, -496772, -496933, -497092, -497251, -497409, -497566, -497722, -497878, -498032, -498185, -498338, -498489, -498640, -498790, -498939, -499087, -499234, -499381, -499526, -499671, -499815, -499958, -500100, -500241, -500381, -500521, -500660, -500798, -500935, -501072, -501207, -501342, -501476, -501610, -501742, -501874, -502005, -502135, -502264, -502393, -502521, -502648, -502774, -502900, -503025, -503149, -503273, -503396, -503518, -503639, -503760, -503880, -503999, -504117, -504235, -504352, -504469, -504585, -504700, -504814, -504928, -505041, -505153, -505265, -505376, -505487, -505597, -505706, -505814, -505922, -506030, -506136, -506242, -506348, -506453, -506557, -506660, -506763, -506866, -506968, -507069, -507169, -507269, -507369, -507468, -507566, -507664, -507761, -507857, -507953, -508049, -508144, -508238, -508332, -508425, -508518, -508610, -508701, -508792, -508883, -508973, -509062, -509151, -509240, -509328, -509415, -509502, -509588, -509674, -509760, -509845, -509929, -510013, -510096, -510179, -510262, -510343, -510425, -510506, -510586, -510666, -510746, -510825, -510904, -510982, -511060, -511137, -511214, -511290, -511366, -511442, -511517, -511591, -511666, -511739, -511813, -511885, -511958, -512030, -512102, -512173, -512244, -512314, -512384, -512453, -512523, -512591, -512660, -512728, -512795, -512862, -512929, -512995, -513061, -513127, -513192, -513257, -513321, -513385, -513449, -513512, -513575, -513638, -513700, -513762, -513824, -513885, -513946, -514006, -514066, -514126, -514185, -514244, -514303, -514361, -514419, -514477, -514534, -514591, -514648, -514704, -514760, -514816, -514871, -514926, -514981, -515035, -515089, -515143, -515196, -515249, -515302, -515355, -515407, -515459, -515510, -515562, -515613, -515663, -515714, -515764, -515814, -515863, -515912, -515961, -516010, -516058, -516106, -516154, -516202, -516249, -516296, -516343, -516389, -516435, -516481, -516527, -516572, -516617, -516662, -516706, -516751, -516795, -516839, -516882, -516925, -516968, -517011, -517054, -517096, -517138, -517180, -517221, -517263, -517304, -517344, -517385, -517425, -517465, -517505, -517545, -517584, -517623, -517662, -517701, -517740, -517778, -517816, -517854, -517891, -517929, -517966, -518003, -518039, -518076, -518112, -518148, -518184, -518220, -518255, -518291, -518326, -518360, -518395, -518429, -518464, -518498, -518532, -518565, -518599, -518632, -518665, -518698, -518730, -518763, -518795, -518827, -518859, -518891, -518922, -518954, -518985, -519016, -519047, -519077, -519108, -519138, -519168, -519198, -519228, -519257, -519287, -519316, -519345, -519374, -519403, -519431, -519459, -519488, -519516, -519544, -519571, -519599, -519626, -519654, -519681, -519708, -519734, -519761, -519787, -519814, -519840, -519866, -519892, -519917, -519943, -519968, -519993, -520019, -520043, -520068, -520093, -520117, -520142, -520166, -520190, -520214, -520238, -520261, -520285, -520308, -520332, -520355, -520378, -520401, -520423, -520446, -520468, -520491, -520513, -520535, -520557, -520579, -520600, -520622, -520643, -520665, -520686, -520707, -520728, -520748, -520769, -520790, -520810, -520830, -520851, -520871, -520891, -520911, -520930, -520950, -520969, -520989, -521008, -521027, -521046, -521065, -521084, -521103, -521121, -521140, -521158, -521177, -521195, -521213, -521231, -521249, -521266, -521284, -521302, -521319, -521336, -521354, -521371, -521388, -521405, -521422, -521438, -521455, -521472, -521488, -521504, -521521, -521537, -521553, -521569, -521585, -521601, -521616, -521632, -521647, -521663, -521678, -521693, -521709, -521724, -521739, -521753, -521768, -521783, -521798, -521812, -521827, -521841, -521855, -521870, -521884, -521898, -521912, -521926, -521939, -521953, -521967, -521980, -521994, -522007, -522020, -522034, -522047, -522060, -522073, -522086, -522099, -522112, -522124, -522137, -522150, -522162, -522174, -522187, -522199, -522211, -522223, -522235, -522247, -522259, -522271, -522283, -522295, -522306, -522318, -522329, -522341, -522352, -522364, -522375, -522386, -522397, -522408, -522419, -522430, -522441, -522452, -522462, -522473, -522484, -522494, -522505, -522515, -522526, -522536, -522546, -522556, -522566, -522576, -522586, -522596, -522606, -522616, -522626, -522636, -522645, -522655, -522664, -522674, -522683, -522693, -522702, -522711, -522720, -522730, -522739, -522748, -522757, -522766, -522775, -522783, -522792, -522801, -522810, -522818, -522827, -522835, -522844, -522852, -522861, -522869, -522877, -522886, -522894, -522902, -522910, -522918, -522926, -522934, -522942, -522950, -522958, -522965, -522973, -522981};
constexpr qaint Sigmoid_table[1024] = {262144, 262912, 263680, 264448, 265216, 265984, 266752, 267519, 268287, 269054, 269822, 270589, 271356, 272123, 272890, 273657, 274423, 275189, 275955, 276721, 277486, 278252, 279017, 279781, 280546, 281310, 282073, 282837, 283600, 284363, 285125, 285887, 286648, 287409, 288170, 288930, 289690, 290449, 291208, 291966, 292724, 293481, 294238, 294994, 295750, 296505, 297260, 298014, 298767, 299520, 300272, 301023, 301774, 302524, 303273, 304022, 304770, 305518, 306264, 307010, 307755, 308500, 309243, 309986, 310728, 311469, 312210, 312949, 313688, 314426, 315163, 315899, 316634, 317369, 318102, 318835, 319566, 320297, 321027, 321755, 322483, 323210, 323936, 324661, 325384, 326107, 326829, 327550, 328269, 328988, 329706, 330422, 331137, 331852, 332565, 333277, 333988, 334697, 335406, 336113, 336820, 337525, 338229, 338931, 339633, 340333, 341032, 341730, 342427, 343122, 343816, 344509, 345200, 345891, 346580, 347267, 347954, 348639, 349322, 350005, 350686, 351366, 352044, 352721, 353397, 354071, 354744, 355415, 356085, 356754, 357421, 358087, 358751, 359414, 360076, 360736, 361395, 362052, 362708, 363362, 364015, 364666, 365316, 365964, 366611, 367256, 367900, 368542, 369183, 369822, 370460, 371096, 371730, 372363, 372995, 373625, 374253, 374880, 375505, 376128, 376750, 377371, 377990, 378607, 379222, 379836, 380449, 381060, 381669, 382276, 382882, 383486, 384089, 384690, 385289, 385887, 386483, 387078, 387670, 388261, 388851, 389439, 390025, 390609, 391192, 391773, 392352, 392930, 393506, 394080, 394653, 395224, 395793, 396360, 396926, 397490, 398053, 398613, 399172, 399730, 400285, 400839, 401391, 401942, 402491, 403038, 403583, 404126, 404668, 405208, 405747, 406283, 406818, 407352, 407883, 408413, 408941, 409467, 409992, 410515, 411036, 411555, 412073, 412589, 413103, 413615, 414126, 414635, 415142, 415648, 416152, 416654, 417154, 417653, 418149, 418645, 419138, 419630, 420120, 420608, 421094, 421579, 422062, 422544, 423023, 423501, 423977, 424452, 424924, 425395, 425865, 426332, 426798, 427262, 427725, 428185, 428644, 429102, 429557, 430011, 430464, 430914, 431363, 431810, 432255, 432699, 433141, 433582, 434020, 434457, 434893, 435326, 435758, 436189, 436617, 437044, 437470, 437893, 438315, 438735, 439154, 439571, 439986, 440400, 440812, 441223, 441631, 442039, 442444, 442848, 443250, 443651, 444050, 444447, 444843, 445237, 445630, 446021, 446410, 446798, 447184, 447568, 447951, 448333, 448712, 449091, 449467, 449842, 450216, 450588, 450958, 451327, 451694, 452060, 452424, 452787, 453148, 453507, 453865, 454222, 454576, 454930, 455282, 455632, 455981, 456328, 456674, 457018, 457361, 457703, 458042, 458381, 458718, 459053, 459387, 459720, 460051, 460380, 460708, 461035, 461360, 461684, 462006, 462327, 462646, 462964, 463281, 463596, 463910, 464222, 464533, 464843, 465151, 465457, 465763, 466067, 466369, 466670, 466970, 467269, 467566, 467862, 468156, 468449, 468741, 469031, 469320, 469608, 469894, 470179, 470462, 470745, 471026, 471306, 471584, 471861, 472137, 472411, 472685, 472957, 473227, 473497, 473765, 474032, 474297, 474562, 474825, 475087, 475347, 475607, 475865, 476122, 476377, 476632, 476885, 477137, 477388, 477638, 477886, 478133, 478379, 478624, 478868, 479110, 479352, 479592, 479831, 480069, 480305, 480541, 480775, 481008, 481240, 481471, 481701, 481930, 482158, 482384, 482609, 482834, 483057, 483279, 483500, 483720, 483938, 484156, 484373, 484588, 484803, 485016, 485228, 485440, 485650, 485859, 486067, 486274, 486480, 486685, 486889, 487092, 487294, 487495, 487695, 487894, 488092, 488289, 488485, 488680, 488874, 489067, 489259, 489450, 489640, 489829, 490018, 490205, 490391, 490576, 490761, 490944, 491127, 491308, 491489, 491668, 491847, 492025, 492202, 492378, 492553, 492727, 492901, 493073, 493245, 493415, 493585, 493754, 493922, 494089, 494256, 494421, 494586, 494750, 494912, 495075, 495236, 495396, 495556, 495714, 495872, 496029, 496186, 496341, 496496, 496649, 496802, 496955, 497106, 497257, 497406, 497556, 497704, 497851, 497998, 498144, 498289, 498433, 498577, 498720, 498862, 499004, 499144, 499284, 499423, 499562, 499699, 499836, 499972, 500108, 500243, 500377, 500510, 500643, 500775, 500906, 501037, 501166, 501296, 501424, 501552, 501679, 501805, 501931, 502056, 502181, 502304, 502427, 502550, 502672, 502793, 502913, 503033, 503152, 503271, 503389, 503506, 503622, 503738, 503854, 503969, 504083, 504196, 504309, 504421, 504533, 504644, 504755, 504865, 504974, 505083, 505191, 505298, 505405, 505512, 505617, 505723, 505827, 505931, 506035, 506138, 506240, 506342, 506443, 506544, 506644, 506744, 506843, 506941, 507039, 507137, 507234, 507330, 507426, 507522, 507616, 507711, 507804, 507898, 507991, 508083, 508175, 508266, 508357, 508447, 508537, 508626, 508715, 508803, 508891, 508978, 509065, 509151, 509237, 509323, 509408, 509492, 509576, 509660, 509743, 509825, 509908, 509989, 510071, 510151, 510232, 510312, 510391, 510470, 510549, 510627, 510705, 510782, 510859, 510935, 511011, 511087, 511162, 511237, 511311, 511385, 511459, 511532, 511605, 511677, 511749, 511820, 511892, 511962, 512033, 512103, 512172, 512241, 512310, 512378, 512446, 512514, 512581, 512648, 512715, 512781, 512847, 512912, 512977, 513042, 513106, 513170, 513233, 513297, 513360, 513422, 513484, 513546, 513608, 513669, 513729, 513790, 513850, 513910, 513969, 514028, 514087, 514146, 514204, 514261, 514319, 514376, 514433, 514489, 514546, 514601, 514657, 514712, 514767, 514822, 514876, 514930, 514984, 515037, 515090, 515143, 515196, 515248, 515300, 515351, 515403, 515454, 515504, 515555, 515605, 515655, 515705, 515754, 515803, 515852, 515900, 515948, 515996, 516044, 516092, 516139, 516186, 516232, 516278, 516325, 516370, 516416, 516461, 516506, 516551, 516596, 516640, 516684, 516728, 516771, 516815, 516858, 516900, 516943, 516985, 517027, 517069, 517111, 517152, 517193, 517234, 517275, 517315, 517355, 517395, 517435, 517475, 517514, 517553, 517592, 517630, 517669, 517707, 517745, 517783, 517820, 517858, 517895, 517932, 517968, 518005, 518041, 518077, 518113, 518149, 518184, 518219, 518254, 518289, 518324, 518358, 518393, 518427, 518461, 518494, 518528, 518561, 518594, 518627, 518660, 518692, 518725, 518757, 518789, 518820, 518852, 518884, 518915, 518946, 518977, 519007, 519038, 519068, 519099, 519129, 519158, 519188, 519218, 519247, 519276, 519305, 519334, 519363, 519391, 519419, 519448, 519476, 519504, 519531, 519559, 519586, 519613, 519640, 519667, 519694, 519721, 519747, 519773, 519800, 519826, 519851, 519877, 519903, 519928, 519953, 519979, 520003, 520028, 520053, 520078, 520102, 520126, 520150, 520174, 520198, 520222, 520245, 520269, 520292, 520315, 520338, 520361, 520384, 520407, 520429, 520452, 520474, 520496, 520518, 520540, 520562, 520583, 520605, 520626, 520647, 520668, 520689, 520710, 520731, 520752, 520772, 520793, 520813, 520833, 520853, 520873, 520893, 520913, 520932, 520952, 520971, 520990, 521009, 521028, 521047, 521066, 521085, 521104, 521122, 521140, 521159, 521177, 521195, 521213, 521231, 521248, 521266, 521284, 521301, 521319, 521336, 521353, 521370, 521387, 521404, 521421, 521437, 521454, 521470, 521487, 521503, 521519, 521535, 521551, 521567, 521583, 521599, 521614, 521630, 521645, 521661, 521676, 521691, 521706, 521721, 521736, 521751, 521766, 521780, 521795, 521809, 521824, 521838, 521852, 521867, 521881, 521895, 521909, 521922, 521936, 521950, 521963, 521977, 521990, 522004, 522017, 522030, 522043, 522056, 522069, 522082, 522095, 522108, 522121, 522133, 522146, 522158, 522171, 522183, 522195, 522207, 522219, 522232, 522243, 522255, 522267, 522279, 522291, 522302, 522314, 522325, 522337, 522348, 522359, 522371, 522382, 522393, 522404, 522415, 522426, 522437, 522447, 522458, 522469, 522479, 522490, 522500, 522511, 522521, 522531, 522542, 522552, 522562, 522572, 522582, 522592, 522602, 522612, 522621, 522631, 522641, 522650, 522660, 522669, 522679, 522688, 522697, 522707, 522716, 522725, 522734, 522743, 522752, 522761, 522770, 522779, 522788, 522796, 522805, 522814, 522822, 522831, 522839, 522848, 522856, 522865, 522873, 522881, 522889, 522897, 522906, 522914, 522922, 522930, 522937, 522945, 522953, 522961, 522969, 522976, 522984};

void ReLU(qaint* x, const int channels, const int height, const int width) {
    const int xshift = oin_shifts[other_cnt];
    const int yshift = oout_shifts[other_cnt];
    print_neg_shift("relu", "xshift", xshift);
    print_neg_shift("relu", "yshift", yshift);
    print_neg_shift("relu", "yshift - xshift", yshift - xshift);
    other_cnt++;
    for (int idx = 0; idx < channels * height * width; idx++)
        x[idx] = max(0, x[idx]) << (yshift - xshift);
}


constexpr int celushifts[2] = {17, 17};
int celu_cnt = 0;

void celu(qaint* x, const int channels, const int height, const int width) {
    const int xshift = celushifts[celu_cnt];
    celu_cnt = 1 - celu_cnt;
    for (int idx = 0; idx < channels * height * width; idx++) {
        const float xx = x[idx] / (float) (1 << xshift);
        x[idx] = max(0, x[idx] << (actbit - xshift)) + min(0, (int) ((exp(xx) - 1) * (1 << actbit)));
    }
}


void Sigmoid(qaint* x, const int channels, const int height, const int width) {
    const int xshift = cout_shifts[conv_cnt-1];
    // if (xshift < tbbit) print2("xshift < tbbit:", xshift);

    for (int idx = 0; idx < channels * height * width; idx++) {
        const float xx = x[idx] / (float) (1 << xshift);
        x[idx] = 1.0 / (1.0 + exp(-xx)) * (1 << actbit);
    }
}
