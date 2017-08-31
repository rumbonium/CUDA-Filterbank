//
//  filtd.h
//  
//
//  Created by Photorithm on 6/12/17.
//
//

#ifndef filtd_h
#define filtd_h
/*
 *
 * filter for differentiation
 *
 */
const int CDL = 41;
const float CD[CDL] = {
    -0.0040000000000000009506285,
    0.0045085980698019280724087,
    -0.0056952223613460761092453,
    0.0076551175807853403046388,
    -0.0104907614117202632741943,
    0.0143153920436125438320207,
    -0.0192584845675330268433001,
    0.0254741823169083447808703,
    -0.0331543485489603495519617,
    0.0425491041892267080060108,
    -0.0540000000000000063282712,
    0.0679955393242784728036199,
    -0.0852684771765594706760538,
    0.1069765185543130742162887,
    -0.1350635360090896019968909,
    0.1730538238691623764697880,
    -0.2280369543531189369112155,
    0.3166210003755497437438748,
    -0.4887429987478853488092057,
    0.9943366366737633743611013,
    0.0000000000000000000000000,
    -0.9943366366737633743611013,
    0.4887429987478853488092057,
    -0.3166210003755497437438748,
    0.2280369543531189369112155,
    -0.1730538238691623764697880,
    0.1350635360090896019968909,
    -0.1069765185543130742162887,
    0.0852684771765594706760538,
    -0.0679955393242784728036199,
    0.0540000000000000063282712,
    -0.0425491041892267080060108,
    0.0331543485489603495519617,
    -0.0254741823169083447808703,
    0.0192584845675330268433001,
    -0.0143153920436125438320207,
    0.0104907614117202632741943,
    -0.0076551175807853403046388,
    0.0056952223613460761092453,
    -0.0045085980698019280724087,
    0.0040000000000000009506285
};

#endif /* filtd_h */
