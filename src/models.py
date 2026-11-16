import numpy as np
from surprise import Dataset,Reader,KNNBasic,SVD,NMF

class Popularity:
    def fit(self,a):
        self.pop=a.groupby('item').affinity.mean().reindex(range(96),fill_value=0).to_numpy(); return self
    def predict(self,user,items): return self.pop[np.asarray(items,dtype=int)]

class PooledPopularity(Popularity):
    def fit(self,a):
        a=a.copy(); a['item']=a.item%48; p=a.groupby('item').affinity.mean().reindex(range(48),fill_value=0).to_numpy(); self.pop=np.tile(p,2); return self

class SurpriseModel(Popularity):
    def __init__(self,name='SVD',seed=11,**params): self.name=name; self.seed=seed; self.params=params
    def fit(self,a):
        super().fit(a); self.users=set(a.household_id)
        data=Dataset.load_from_df(a[['household_id','item','affinity']],Reader(rating_scale=(0,1))).build_full_trainset()
        if self.name in ['UserKNN','ItemKNN']:
            self.model=KNNBasic(sim_options={'name':'cosine','user_based':self.name=='UserKNN'},verbose=False,**self.params)
        elif self.name=='SVD': self.model=SVD(random_state=self.seed,**self.params)
        elif self.name=='NMF': self.model=NMF(random_state=self.seed,**self.params)
        else: raise ValueError(self.name)
        self.model.fit(data); return self
    def predict(self,user,items):
        if user not in self.users: return super().predict(user,items)
        return np.array([self.model.predict(user,int(i)).est for i in items])

class ImplicitALS(Popularity):
    """Hu-style all-pairs confidence-weighted implicit objective.
    Observed p=1[affinity>=0.5] and c=confidence, unobserved p=0,c=1.
    Each alternating solve includes all items/users via Gram matrices.
    """
    def __init__(self,factors=12,regularization=.1,iterations=12,seed=11,preference_threshold=.5):
        self.preference_threshold=preference_threshold; self.factors=factors; self.regularization=regularization; self.iterations=iterations; self.seed=seed
    def fit(self,a):
        super().fit(a); self.users={u:i for i,u in enumerate(sorted(a.household_id.unique()))}; n=len(self.users)
        p=np.zeros((n,96)); c=np.ones((n,96))
        for r in a.itertuples(): p[self.users[r.household_id],r.item]=float(r.affinity>=self.preference_threshold); c[self.users[r.household_id],r.item]=r.confidence
        rng=np.random.default_rng(self.seed); self.X=rng.normal(0,.1,(n,self.factors)); self.Y=rng.normal(0,.1,(96,self.factors)); eye=np.eye(self.factors)*self.regularization
        for _ in range(self.iterations):
            for target,fixed,pp,cc in [(self.X,self.Y,p,c),(self.Y,self.X,p.T,c.T)]:
                gram=fixed.T@fixed
                for i in range(len(target)):
                    target[i]=np.linalg.solve(gram+fixed.T@((cc[i]-1)[:,None]*fixed)+eye,fixed.T@(cc[i]*pp[i]))
        return self
    def predict(self,user,items):
        if user not in self.users: return super().predict(user,items)
        return np.clip(self.Y[np.asarray(items)]@self.X[self.users[user]],0,1)

def make_model(name,seed=11,params=None):
    if name=='Popularity': return Popularity()
    if name=='PooledPopularity': return PooledPopularity()
    if name=='PooledImplicitALS': return PooledImplicitALS(seed=seed,**(params or {}))
    if name=='ImplicitALS': return ImplicitALS(seed=seed,**(params or {}))
    return SurpriseModel(name,seed,**(params or {}))

class PooledImplicitALS(ImplicitALS):
    """Same implicit solver, weekday/weekend signals pooled into 48 slots."""
    def fit(self,a):
        pooled=a.assign(item=a.item%48).groupby(['household_id','item'],as_index=False).agg(affinity=('affinity','mean'),confidence=('confidence','mean'))
        return super().fit(pooled)
    def predict(self,user,items): return super().predict(user,np.asarray(items)%48)
