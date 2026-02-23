from data_provider.data_loader import DrivAerML, NASA, AirCraft, DTCHull, Car_Crash


def get_data(args, full_mesh=False):
    data_dict = {
        'DrivAerML': DrivAerML,
        'NASA': NASA,
        'AirCraft': AirCraft,
        'DTCHull': DTCHull,
        'Car_Crash': Car_Crash,
    }
    dataset = data_dict[args.loader](args)
    train_loader, test_loader, shapelist = dataset.get_loader(full_mesh)
    return dataset, train_loader, test_loader, shapelist
