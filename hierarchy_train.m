function [ model ] = hierarchy_train( feature , label )
    [ feature_count , dimension ] = size( feature ) ;
    label_count = max( label ) ;
    LAMBDA = 0.0001 ;

    model = create_hierarchy_node( 1 : label_count , label_count , dimension ) ;
    now_pt = 0 ;
    len_pt = 1 ;
    node = ones( feature_count , 1 ) ;
    while ( now_pt < len_pt )
        now_pt = now_pt + 1 ;
        if ( sum( model( now_pt ).l ) == 1 )
            continue ;
        end
        list = find( node == now_pt ) ;
        sub_feature = feature( list , : ) ;
        sub_label = label( list ) ;
        label_list = unique( sub_label ) ;
        cluster = kmeans( sub_feature , 2 ) ;
        
        % Calc Q
        valq = zeros( length( label_list ) , 1 ) ;
        for i = 1 : length( label_list ) 
            c = label_list( i ) ;
            valq( i ) = ( sum( and( cluster == 1 , sub_label == c ) ) - ...
                sum( and( cluster == 2 , sub_label == c ) ) ) / sum( sub_label == c ) ;
        end
        
        % Training Decision Boundary
        positive_label = label_list( valq >= 0.8 ) ;
        negative_label = label_list( valq <= -0.8 ) ;
        ignore_label   = label_list( and( valq > -0.8 , valq < 0.8 ) ) ;
        disp( sprintf( 'Node %d: +1(%d) -1(%d) 0(%d)' , now_pt , ...
            length( positive_label ) , length( negative_label ) , length( ignore_label ) ) ) ;
        if ( isempty( positive_label ) || isempty( negative_label ) )
            % Training One-vs-All
            positive_label = [ label_list( 1 ) ] ;
            negative_label = label_list( 2 : end ) ;
            ignore_label = [] ;
        end

        % Training Relaxed Hierarchy
        positive_set = find( ismember( sub_label , positive_label ) ) ;
        negative_set = find( ismember( sub_label , negative_label ) ) ;
        train_feature = sub_feature( [ positive_set ; negative_set ] , : ) ;
        train_label =  ones( length( [ positive_set ; negative_set ] ) , 1 ) ;
        train_label( length( positive_set ) + 1 : end ) = -1 ;
        [ model( now_pt ).w , model( now_pt ).b , ~ ] = vl_svmtrain( train_feature' , train_label , LAMBDA ) ;
        ignore_list = find( ismember( sub_label , ignore_label ) ) ;
        ignore_feature = sub_feature( ignore_list , : ) ;
        esti = ignore_feature * model( now_pt ).w + model( now_pt ).b ;

        % Build DAG Struct
        left_label  = union( positive_label , sub_label( ignore_list( esti >= 0 ) ) ) ; % 1 
        right_label = union( negative_label , sub_label( ignore_list( esti <  0 ) ) ) ; % -1 
        for i = now_pt + 1 : len_pt
            if ( isequal( model( i ).label_list , left_label ) )
                model( now_pt ).next( 1 ) = i ;
            end
            if ( isequal( model( i ).label_list , right_label ) )
                model( now_pt ).next( 2 ) = i ;
            end
        end
        if ( model( now_pt ).next( 1 ) == 0 )
            len_pt = len_pt + 1 ;
            model( now_pt ).next( 1 ) = len_pt ;
            model = [ model , create_hierarchy_node( left_label , label_count , dimension ) ] ;
        end
        if ( model( now_pt ).next( 2 ) == 0 )
            len_pt = len_pt + 1 ;
            model( now_pt ).next( 2 ) = len_pt ;
            model = [ model , create_hierarchy_node( right_label , label_count , dimension ) ] ;
        end
        node( list( ismember( sub_label , positive_label ) ) ) = model( now_pt ).next( 1 ) ;
        node( list( ismember( sub_label , negative_label ) ) ) = model( now_pt ).next( 2 ) ;
        node( list( ignore_list( esti > 0 ) ) ) = model( now_pt ).next( 1 ) ;
        node( list( ignore_list( esti < 0 ) ) ) = model( now_pt ).next( 2 ) ;
    end
end
