graph [
  node [
    id 0
    label "ATLAM5"
    x -84.3833
    y 33.75
    weight 1343
  ]
  node [
    id 1
    label "ATLAng"
    x -85.5
    y 34.5
    weight 1315
  ]
  node [
    id 2
    label "CHINng"
    x -87.6167
    y 41.8333
    weight 1379
  ]
  node [
    id 3
    label "DNVRng"
    x -105.0
    y 40.75
    weight 1304
  ]
  node [
    id 4
    label "HSTNng"
    x -95.517364
    y 29.770031
    weight 1031
  ]
  node [
    id 5
    label "IPLSng"
    x -86.159535
    y 39.780622
    weight 1090
  ]
  node [
    id 6
    label "KSCYng"
    x -96.596704
    y 38.961694
    weight 1331
  ]
  node [
    id 7
    label "LOSAng"
    x -118.25
    y 34.05
    weight 1096
  ]
  node [
    id 8
    label "NYCMng"
    x -73.9667
    y 40.7833
    weight 1332
  ]
  node [
    id 9
    label "SNVAng"
    x -122.02553
    y 37.38575
    weight 1309
  ]
  node [
    id 10
    label "STTLng"
    x -122.3
    y 47.6
    weight 1358
  ]
  node [
    id 11
    label "WASHng"
    x -77.026842
    y 38.897303
    weight 1363
  ]
  edge [
    source 0
    target 1
    id "ATLAM5_ATLAng"
    capacity 40000.0
    cost 133.0
    weight 12
  ]
  edge [
    source 1
    target 4
    id "ATLAng_HSTNng"
    capacity 40000.0
    cost 1081.0
    weight 12
  ]
  edge [
    source 1
    target 5
    id "ATLAng_IPLSng"
    capacity 40000.0
    cost 591.0
    weight 10
  ]
  edge [
    source 1
    target 11
    id "ATLAng_WASHng"
    capacity 40000.0
    cost 901.0
    weight 14
  ]
  edge [
    source 2
    target 5
    id "CHINng_IPLSng"
    capacity 40000.0
    cost 260.0
    weight 13
  ]
  edge [
    source 2
    target 8
    id "CHINng_NYCMng"
    capacity 40000.0
    cost 1147.0
    weight 15
  ]
  edge [
    source 3
    target 6
    id "DNVRng_KSCYng"
    capacity 40000.0
    cost 745.0
    weight 13
  ]
  edge [
    source 3
    target 9
    id "DNVRng_SNVAng"
    capacity 40000.0
    cost 1516.0
    weight 10
  ]
  edge [
    source 3
    target 10
    id "DNVRng_STTLng"
    capacity 40000.0
    cost 1573.0
    weight 12
  ]
  edge [
    source 4
    target 6
    id "HSTNng_KSCYng"
    capacity 40000.0
    cost 1028.0
    weight 15
  ]
  edge [
    source 4
    target 7
    id "HSTNng_LOSAng"
    capacity 40000.0
    cost 2196.0
    weight 10
  ]
  edge [
    source 5
    target 6
    id "IPLSng_KSCYng"
    capacity 40000.0
    cost 903.0
    weight 12
  ]
  edge [
    source 7
    target 9
    id "LOSAng_SNVAng"
    capacity 40000.0
    cost 505.0
    weight 12
  ]
  edge [
    source 8
    target 11
    id "NYCMng_WASHng"
    capacity 40000.0
    cost 336.0
    weight 15
  ]
  edge [
    source 9
    target 10
    id "SNVAng_STTLng"
    capacity 40000.0
    cost 1138.0
    weight 10
  ]
]
