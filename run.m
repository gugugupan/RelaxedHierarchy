load( 'toy_data.mat' ) ;
model = hierarchy_train( feature , label ) ;
predict_label = hierarchy_test( feature , model ) ;
disp( sum( predict_label == label ) / length( label ) ) ;
