import requests


# read in games from lichess api example is https://lichess.org/api/games/user/NotAHeroinDealer
# with a query parameter for analyzed = true
# returns a list of games
# do it using a stream

def get_games(username):
    url = 'https://lichess.org/api/games/user/' + username
    params = {'analysed': True, 'max': 10, 'evals': True}
    r = requests.get(url, params=params)
    return r.json()

# read in games from lichess api example is https://lichess.org/api/games/user/NotAHeroinDealer
# with a query parameter for analyzed = true
# returns a list of games
# do it using a stream
get_games('NotAHeroinDealer')