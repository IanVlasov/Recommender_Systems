import pandas as pd
import numpy as np

from itertools import chain

# Для работы с матрицами
from scipy.sparse import csr_matrix

from src.utils import create_prices_df

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender, TFIDFRecommender, BM25Recommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    data: pd.DataFrame с данными для обучения модели
    """
    
    def __init__(self, data, take_n_popular=5000, weighting=False, k=0.05, b=0.8):
        
       
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        # prices df
        self.df_prices = create_prices_df(data)
        
        # Дешевые товары для фильтрации
        self.filter_cheap_items = self._get_cheap_items(7)
        
        # item-department dict
        self.item_dep_dict = dict(zip(data.item_id, data.sub_commodity_desc))
        # department-item dict
        self.dep_item_dict = data.groupby('sub_commodity_desc')['item_id'].unique()
        
        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid,\
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        
        # Own recommender обучается до взвешивания матриц
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        if weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T, K1=k, B=b).T 
        
        if weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T 
        
        self.als_model = self.fit_als(self.user_item_matrix)
        
    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
    
    def _recalculate_dicts(self):
        """На случай изменения user_item матрицы
        Обновляет словари
        """
        self.id_to_itemid, self.id_to_userid,\
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        
    def _refit_all_models(self, weighting=False):
        """На случай изменения user_item матрицы
        Переобучает модели own_recommender и als на новые веса
        """
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        if weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        if weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T 
        
        self.als_model = self.fit_als(self.user_item_matrix)
            
    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.als_model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5, filter_department=False,
                                 filter_items=[]):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            if not filter_department:
                i = 0
                while len(recommendations) != N:
                    if self.overall_top_purchases[i] not in set(filter_items):
                        recommendations.append(self.overall_top_purchases[i])
                    i+=1
                
            else:
                filter_items_from_dep = self._get_items_from_departments(recommendations)
                filter_items.extend(filter_items_from_dep)
                    
                canditates = [candidate for candidate in self.overall_top_purchases 
                              if candidate not in set(filter_items)
                             ]
                
                additional_recommendations = self._choose_from_different_departments(canditates, N=N)
                recommendations.extend(additional_recommendations)
                recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5, filter_already_liked_items=False,
                             filter_department=False, filter_items=[]
                            ):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = []
        
        try:
            filter_items_als = [self.itemid_to_id[item] for item in filter_items]
            filter_items_als.append(self.itemid_to_id[999999])
            
            if not filter_department:
                res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                N=N,
                                                filter_already_liked_items=filter_already_liked_items,
                                                filter_items=filter_items_als,
                                                recalculate_user=True)]
            
            else:
                canditates = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                            user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                            N=5*N,
                                            filter_already_liked_items=filter_already_liked_items,
                                            filter_items=filter_items_als,
                                            recalculate_user=True)]
                
                res = self._choose_from_different_departments(canditates, N=N)
                
                
        except IndexError:
            pass

        res = self._extend_with_top_popular(res, N=N, filter_department=filter_department, filter_items=filter_items)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def _choose_from_different_departments(self, candidates, N=5):
        
        res = []
        departments = set()

        for item in candidates:
            if self.item_dep_dict[item] in departments:
                continue
            else:
                res.append(item)
                departments.add(self.item_dep_dict[item])
                if len(res) == N:
                    break
            
        return res
    
    def _get_items_from_departments(self, items):
        already_exist_departments = [self.item_dep_dict[item] for item in items]
            
        filter_items = []
        for dep in already_exist_departments:
            filter_items.extend(self.dep_item_dict[dep].tolist()) 
            
        return filter_items
    
    def _get_cheap_items(self, thresh):
        cheap_items = self.df_prices.loc[self.df_prices['price'] < thresh].index.tolist()
        return cheap_items
    
    def _get_expensive_items(self, thresh):
        expensive_items = self.df_prices.loc[self.df_prices['price'] > thresh].index.tolist()
        return expensive_items
    
    def _test_final_recommendations(self, user, recommendation):
        """Функция проверяет бизнес ограничения в рекомендациях"""
    
        prices = self.df_prices.loc[recommendation, 'price'].tolist()
        departments = [self.item_dep_dict[item] for item in recommendation]
        new_items = set(recommendation).difference(set(self.top_purchases\
                                                       .loc[self.top_purchases['user_id'] == user,'item_id'].tolist()))
        
        assert len(recommendation) == 5, f'Recommendation list has length less than 5 elements for user {recommendation}'
        assert len(set(departments)) == 5, 'business constraint for items from different department is failed '\
                                           f'for user {user}. Recommendation list is {recommendation}'
        assert len(new_items) >= 2, 'business constraint for 2 new items is failed '\
                                    f'for user {user}. Recommendation list is {recommendation}'
        assert all(price >= 1 for price in prices), 'business constraint for items less than 1$ is failed '\
                                                    f'for user {user}. Recommendation list is {recommendation}'
        assert any(price >= 7 for price in prices), 'business constraint for items more than 7$ is failed '\
                                                    f'for user {user}. Recommendation list is {recommendation}'
        
    def get_als_recommendations(self, user, N=5, filter_already_liked_items=False, filter_department=False,
                                filter_items = []
                               ):
        """Рекомендации через стардартные библиотеки implicit"""

        return self._get_recommendations(user, model=self.als_model, N=N,
                                         filter_already_liked_items=filter_already_liked_items,
                                         filter_department=filter_department,
                                         filter_items=filter_items
                                        )

    def get_own_recommendations(self, user, N=5, filter_department=False, filter_items=[]):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        return self._get_recommendations(user, model=self.own_recommender, N=N,
                                         filter_department=filter_department, filter_items=filter_items)
    
    def get_custom_own_recommendations(self, user, N=5, filter_department=False, filter_items=[]):
        """Реализация на основе top_purchases"""
        
        res = []
        if not filter_department:
            i = 0
            candidates = self.top_purchases.loc[self.top_purchases['user_id'] == user, 'item_id'].tolist()
            while len(res) != N and i < len(candidates):
                if candidates[i] not in set(filter_items):
                    res.append(candidates[i])
                i+=1
                
        else:
            canditates = [item 
                          for item in self.top_purchases.loc[self.top_purchases['user_id'] == user, 'item_id'].tolist()
                          if item not in set(filter_items)
                         ]
            
            res = self._choose_from_different_departments(canditates, N=N)
        
        res = self._extend_with_top_popular(res, N=N, filter_department=filter_department,
                                                    filter_items=filter_items
                                                   )
        return res
        
    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_n_user_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N).item_id.tolist()
        res = []
        
        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        
        res = self._extend_with_top_popular(res, N=N)        
        res = [id_to_itemid[item] for item in res]
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5, filter_department=False, filter_items=[]):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        res = []
        
        try:
            similar_users = self.als_model.similar_users(self.userid_to_id[user], N=N+1)
            similar_users = [self.id_to_userid[user[0]] for user in similar_users]
            similar_users = similar_users[1:]   # удалим юзера из запроса
            
            filter_add = filter_items
            for user in similar_users:
                
                if res and filter_department:
                    filter_add_dep = self._get_items_from_departments(res)
                    filter_add.extend(filter_add_dep)
                res.extend(self.get_custom_own_recommendations(user, N=1, filter_department=filter_department,
                                                               filter_items=filter_add))
        except IndexError:
            pass

        res = self._extend_with_top_popular(res, N=N, filter_department=filter_department, 
                                            filter_items=filter_items)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_all_users_recommendations(self, method, users, N=5, filter_already_liked_items=False,
                                     filter_department=False
                                    ):
        """Создает рекомендации для всех переданных пользователей выбранным методом"""
        
        methods = {'als': self.get_als_recommendations,
                   'own': self.get_own_recommendations,
                   'custom_own': self.get_custom_own_recommendations,
                   'sim_users': self.get_similar_users_recommendation,
                   'sim_items': self.get_similar_items_recommendation
                  }
        
        assert method in methods.keys(), 'Invalid method'
        
        recommendations = []
        
        if method == 'als':
            for user in users:
                cur_user_recommendation = methods[method](user, N=N, filter_already_liked_items=filter_already_liked_items,
                                                          filter_department=filter_department
                                                         )
                recommendations.append(cur_user_recommendation)
        else:
            for user in users:
                cur_user_recommendation = methods[method](user, N=N, filter_department=filter_department)
                recommendations.append(cur_user_recommendation)
        
        return recommendations
    
    def get_expensive_item_recommendation(self, user):     
        
        expensive_item = 0
        
        if user in self.top_purchases['user_id'].unique():
            user_purchases = self.top_purchases.loc[self.top_purchases['user_id'] == user, 'item_id'].tolist()[:30]
            expensive_items = [item for item in user_purchases if item not in set(self.filter_cheap_items)]
            expensive_item = expensive_items[:1]
        
        if not expensive_item:
            expensive_item = self.get_als_recommendations(user, N=1,
                                                          filter_items=self.filter_cheap_items
                                                         )
#               expensive_item = self.get_similar_users_recommendation(user, N=1,
#                                                                      filter_items=self.filter_cheap_items
#                                                                     )
    
        return expensive_item
    
    def get_final_recommendations(self, users):
        """Финальные рекомендации
        3 рекомендации из собственных покупок + 2 новые из als
        """
        
        res = []
        filter_cheap_items = self._get_cheap_items(2)
        filter_expensive_items = self._get_expensive_items(2)
        
        for user in users:
            recommendation = []
            recommendation.extend(self.get_expensive_item_recommendation(user))
            
            filter_items = self._get_items_from_departments(recommendation) 
            filter_items.extend(filter_cheap_items)
            recommendation.extend(self.get_custom_own_recommendations(user, N=2, filter_department=True,
                                                                        filter_items=filter_items))
            
            filter_items = self._get_items_from_departments(recommendation)
            filter_items.extend(filter_expensive_items)
            
            recommendation.extend(self.get_als_recommendations(user, N=2, filter_department=True,
                                                               filter_already_liked_items=True,
                                                               filter_items=filter_items))
            
            
            self._test_final_recommendations(user, recommendation)
            res.append(recommendation) 
        
        return res
    
    def recalculate_params(self, weighting=False):
        self._recalculate_dicts()
        self._refit_all_models(weighting=weighting)
    
    @staticmethod
    def _prepare_matrix(data):
        """Подготавливает user-item матрицу"""
        
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity', # Можно пробовать другие варианты
                                          aggfunc='count', 
                                          fill_value=0
                                         )

        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix
    
    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
    
    @staticmethod
    def _prepare_item_id_to_ctm(item_features):
        all_ids = item_features['item_id'].tolist()
        ctm_ids = set(item_features[item_features['brand'] == 'Private'].item_id.tolist())
        
        item_id_to_ctm = {}
        
        for item_id in all_ids:
            item_id_to_ctm[item_id] = 1 if item_id in ctm_ids else 0
            
        return item_id_to_ctm
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)
        
        return own_recommender
    
    @staticmethod
    def fit_als(user_item_matrix, n_factors=40, regularization=0.01, iterations=80, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                         regularization=regularization,
                                         iterations=iterations,  
                                         num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)
        
        return model