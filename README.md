# Foveated-YOLT
You Only Look Twice - Foveated version

Para executar thesis.cpp:

./yolt /home/filipa/Documents/Foveated_YOLT/files/ deploy_caffenet.prototxt bvlc_caffenet.caffemodel imagenet_mean.binaryproto val.txt




Objectivo para a tese: 
  - Converter código de Python para C++
  - Comparar os erros de classificação e localização entre o dataset dado e o dataset com blur uniforme
  - Converter imagens to dataset de coordenadas cartesianas para coordenadas polares
  - Testar vários raios de fovea para testar a influência da resolução nas tarefas
