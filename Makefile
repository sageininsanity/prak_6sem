clean:
	rm config.json
	rm meta.json
	rm dq.json
	cp vanilla/* .
	rm data/batches/*
	rm models/MLPRegressor/*
	rm models/PARegressor/*
	rm models/SGDRegressor/*
	rm models/best_model.pkl
	rm summary/*
	rm inference/*

recover:
	cp vanilla/* .