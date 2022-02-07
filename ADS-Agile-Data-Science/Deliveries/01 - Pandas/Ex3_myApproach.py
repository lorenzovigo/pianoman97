# What the teacher said the exercise asked for
def ex3_myApproach():
    # Get the mean rating by genders for each movie and compare
    genderRating = ex2()

    # Keep the highest mean rating by a gender (either M or F) for each movie
    genderMaxRating = genderRating.groupby('title').max()

    # Keep the mean rating by F for each movie, repeating the process in ex2() but only with female ratings
    relevantMovies = data[data['title'].isin(ex1().index)]
    genderFRating = relevantMovies[relevantMovies['gender'] == 'F'].groupby(['title', 'gender'])['rating'].mean().reset_index(level=1, drop=True) # We must get rid of multi-level indices

    # We compare the max gender rating (M or F) with the F rating for each movie
    ratingCompare = pd.concat([genderMaxRating, genderFRating], axis=1)
    ratingCompare.columns = ["MAX", "F"] # To show information better, we may rename the columns

    # When MAX rating equals F rating, it means women rated it better
    result = ratingCompare[ratingCompare["MAX"] == ratingCompare["F"]]
    result.columns = ["MAX", "rating"]

    return result["rating"].sort_values(ascending = False)
ex3_myApproach()