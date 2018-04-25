import Content.Models.TweetToVec as Model
model = Model.TweetToVec()
model.Fit(model.GetFormatedData()[:10000])
