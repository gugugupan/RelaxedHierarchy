function [ st ] = create_hierarchy_node( label_list , label_count , dimension )
    st = struct() ;
    st.l = zeros( label_count , 1 ) ;
    st.l( label_list ) = 1 ;
    st.label_list = label_list ;
    st.next = zeros( 2 , 1 ) ;
    st.w = zeros( dimension , 1 ) ;
    st.b = 0 ;
end

