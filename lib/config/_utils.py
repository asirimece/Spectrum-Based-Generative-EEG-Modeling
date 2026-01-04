import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
from scipy import stats


def resolve_search_list(search: dict):
    d_type = float
    if 'd_type' in search:
        d_type = search['d_type']
    match search['strategy']:
        case 'range':
            decimal_places = len(str(search['step']).split(".")[1]) if "." in str(search['step']) else 0
            values = np.round(np.arange(start=search['start'], stop=search['end'] + search['step'], step=search['step'],
                               dtype=d_type), decimal_places)
            # Required because omegaconf can currently only handle primitive types.
            return values.tolist()
        case 'choice':
            return search['list']
        case 'loguniform':
            if 'size' not in search:
                return stats.loguniform(search['start'], search['end'])
            else:
                return stats.loguniform.rvs(search['start'], search['end'], size=search['size'])
        case 'uniform':
            if 'size' not in search:
                return stats.uniform(search['start'], search['end'])
            else:
                return stats.uniform.rvs(search['start'], search['end'], size=search['size'])
        case 'exponential':
            if 'size' not in search:
                return stats.expon(search['scale'])
            else:
                return stats.expon.rvs(search['scale'], size=search['size'])




def config_to_primitive(config: any):
    return OmegaConf.to_container(config, resolve=True) if type(config) in [DictConfig, ListConfig] else config
