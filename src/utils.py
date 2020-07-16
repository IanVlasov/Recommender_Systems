def prefilter_items(data, item_features=None, take_n_popular=5000, 
                    filters={'top_popular': 0.5, 
                             'top_notpopular': 0.01,
                             'price_less_than': 1,
                             'price_higher_than': 30,
                             'last_purchase_n_weeks_back': 50,
                             'department_filter': []
                            }):
    """Предфильтрация товаров
    
    Input
    ------
    data: pd.DataFrame, 
          Содержит столбцы 'item_id', 'user_id', 'sales_value', 'quantity', 'week_no'
          
    item_features: pd.DataFrame,
                   Содержит фичи товаров. Необходим для фильтрации по категориям. 
    
    take_n_popular: int 
                    Количество возвращаемых товаров
    filters: dict
             словарь, в котором ключами являются названия фильтров, которые необходимо применить, а значениями пороги, 
             по которым необходимо фильтровать. По умолчанию применяются все фильтры, кроме 'department_filter'
             Возможные значения: 'top_popular', 'top_notpopular', 'price_less_than', 'price_higher_than',
                               'last_purchase_n_weeks_back', 'department_filter'
    
    Return
    ------
    data - dataframe, отфильтрованнный в соответствии с указанными filters и содержащий топ-take_n_popular товаров
    """
    # Проверяем, что фильтры указаны верно
    default_filters = set(['top_popular', 'top_notpopular', 'price_less_than', 'price_higher_than', 
                          'last_purchase_n_weeks_back', 'department_filter'])
    
    assert all(filter_ in default_filters for filter_ in filters.keys()), ('Invalid filters\' keys. '
                                                                           'Please check help for '
                                                                           'getting valid values')
    
    # 1. Неинтересные категории
    if filters.get('department_filter'):
        assert item_features is not None, 'Function expects \'item_features\' dataframe for department filter'
        
        # Проверяем, что все фильтруемые категории присутствуют в датасете
        department_mask = [department in set(item_features['department'].unique()) 
                           for department in filters.get('department_filter')]
        not_exist = [filters.get('department_filter')[i] for i in range(len(department_mask))
                    if not department_mask[i]]
        
        assert all(department_mask), 'Following departments in department_filter do not exist in '\
                                    f'item_features dataframe {not_exist}'
        
        # Фильтруем по категориям
        from_filtered_department = item_features[item_features['department'].\
                                                 isin(filters.get('department_filter'))].item_id.tolist()
        data = data[~data['item_id'].isin(from_filtered_department)]
    
    # Фильтрация из урока
    # Подготовка для фильтрации по популярности
    if filters.get('top_popular') is not None or filters.get('top_notpopular') is not None:
        popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
        popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
        
    # Самые популярные (итак купят)
    if filters.get('top_popular') is not None:
        top_popular = popularity[popularity['share_unique_users'] > filters.get('top_popular')].item_id.tolist()
        data = data[~data['item_id'].isin(top_popular)]
    
    # Самые непопулярные (итак не купят)
    if filters.get('top_notpopular') is not None:
        top_notpopular = popularity[popularity['share_unique_users'] < filters.get('top_notpopular')].item_id.tolist()
        data = data[~data['item_id'].isin(top_notpopular)]
    
    # Подготовка для фильтрации по цене
    if filters.get('price_less_than') is not None or filters.get('price_higher_than') is not None:
        data['price'] = data['sales_value'] / data['quantity']
        mean_prices = data.groupby('item_id')['price'].agg('mean').reset_index()
        mean_prices.rename(columns={'price': 'mean_price'}, inplace=True)
        
    # 2. Удаление товаров, со средней ценой < 1$
    if filters.get('price_less_than') is not None:
        low_price = mean_prices[mean_prices['mean_price'] < filters.get('price_less_than')].item_id.tolist()
        data = data[~data['item_id'].isin(low_price)]
    
    # 3. Удаление товаров со соедней ценой > 30$
    if filters.get('price_higher_than') is not None:
        high_price = mean_prices[mean_prices['mean_price'] > filters.get('price_higher_than')].item_id.tolist()
        data = data[~data['item_id'].isin(high_price)]
    
    # 4. Товары, которые не продавались 50 недель
    if filters.get('last_purchase_n_weeks_back') is not None:
        n_weeks_back = data['week_no'].unique().max() - filters.get('last_purchase_n_weeks_back')
        item_ids_before = set(data[data['week_no'] < n_weeks_back].item_id.tolist())
        item_ids_after = set(data[data['week_no'] >= n_weeks_back].item_id.tolist())
        item_ids_last_purchase_n_weeks_back = list(item_ids_before - item_ids_after)
        data = data[~data['item_id'].isin(item_ids_last_purchase_n_weeks_back)]
    
    # 5. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_n = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    data = data[data['item_id'].isin(top_n)]
    
    return data