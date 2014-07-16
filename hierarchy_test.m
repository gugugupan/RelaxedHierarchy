function [ label ] = hierarchy_test( feature , model )
    [ feature_count , ~ ] = size( feature ) ;
    node_count = length( model ) ;
    
    label = zeros( feature_count , 1 ) ;
    node = ones( feature_count , 1 ) ;
    for i = 1 : node_count 
        list = find( node == i ) ;
        if ( sum( model( i ).l ) == 1 )
            label( list ) = find( model( i ).l ) ;
            continue ;
        end
        esti = feature( list , : ) * model( i ).w + model( i ).b ;
        node( list( esti >= 0 ) ) = model( i ).next( 1 ) ;
        node( list( esti < 0  ) ) = model( i ).next( 2 ) ;
    end
end

