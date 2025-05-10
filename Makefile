clean:
	rm -f config.json
	rm -f meta.json
	rm -f dq.json
	cp vanilla/* .
	rm -f training.log
	rm -f data/batches/*
	rm -f models/MLPRegressor/*
	rm -f models/PARegressor/*
	rm -f models/SGDRegressor/*
	rm -f models/best_model.pkl
	rm -f summary/*
	rm -f inference/*