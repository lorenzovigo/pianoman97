Grade: 8

Feedback:
Exercise 1:

Number of iterations and warmup are too large.

fit <- sampling(stanDso, data = dat, iter = 20000, chains=1, warmup = 10000, thin = 1) 

You say: "Sadly, we could not achieve cropping the x axis of the second plot". Why, just add something like 'xlim=c(0,1000),' RTFM (Read The Friendly Manual).

About 'chains = 5': it is too large. Apparently you do not grasp the meaning of this parameter. One uses three or four chains to prevent a possible bias due to starting point and to be able to perform convergence diagnostics.

Exercise 2:

Nice the derivation of the variance of a mixture. Should have been signalled explicitly asa theorem though, either claiming originality or pointing at its source.  I have seen this derivation of the theoretical variance of a mixture pdf in other homeworks. Obviously not all of them are original. not all of them are original. Fair conduct should require acknowledgment of  source and proper reference.

To generalize the Stan code to a three-component mixture a possible solution is a direct extension of the two-component:

theta[u<cumgamma1? 1: (u<cumgamma2 ? 2 : 3)] ;

where cumgamma1 and 2 are the cumulative probs.

Still another possibility is to define a function. This is valid for mixtures of any number of priors.  The mixture coefficients prior.gamma are entered through its cumulative sums cumsum.prior.gamma. A 'virtual' (unwritten) integer parameter j taking values 1:length(prior.gamma) with probabilities vector prior.gamma is created with the following pseudo-code (actually R syntax):

 j<-0

 for (i in 1:length(cumsum.prior.gamma)){

     if (u < cumsum.prior.gamma[i]){

           j <- i              # (with probability prior.gamma[i])

           break

            }

        }

This function, translated into Stan syntax, is written in the functions{} block of the Stan program.

int fake_int_param(real u, vector cg){

       int j ;

      for (i in 1:num_elements(cg)){

             if(u < cg[i])  j = i ;

              }

      return j ;

      }