Grade: 8.5

Feedback:

General assessment:  A nice idea that of comparing different PPL. Also I appreciate that you have tackled two languages, Tensorflow Probability and Edward, which were not covered in our course. By the way, why you are not including Stan in your repertory of PPL's?. 

I feel, however, that this idea is not explored to any significant depth. From your submitted project report, one just learns that languages such as TfP and Edward do exist and, as advertised in their respective home pages, some of their salient features are mentioned. 

I miss both a discussion of syntax and expressivity issues under different languages and an evaluation of compilation/runtime execution/sampling times.  We have used in class this very example to highlight the different capabilities of JAGS and Stan as to categorical parameters and consequences thereof. I would have expected at least a mention of this fact. Also this particular problem is rather elementary, which in principle is not a great disadvantage but my sensation is, as stated in the previous paragraph, it is treated in a somewhat hasty manner. 

Finally, about choosing a PPL. You do not explain motives for your particular choice. Then, you do not pursue the comparison any further.  My take on this subject is as follows:  'TfP' and 'Greta' belong in the main course syllabus (occasionally they did). Due to time constraints they went off-schedule in the current edition of the course. 'Edward' is a different problem. The main GitHub repository seems frozen since about four years ago and its relation to the 'Edward2' dialect is not clear in the available documentation.

Remarks:

In the TfP notebook, for some reason I could  not identify, histograms do not appear correctly (I tried on two different computers). Since Edward is using TfP it inherits this same problem: Either execution gets stuck on 'plt.hist()'  or, after a long time, a weird mangled picture appears. I could circumvent this by explicitly casting the Tf object as the underlying np.array. E.g., 

'thetas_m_heads_array=np.asarray(thetas_m_heads)'

Then everything is fine and works as in your submitted notebooks. Don't know if in your computer  you have either a different version of TfP or matplotlib or you have in place some undocumented configuration quirk taking care of this dissonance by doing this casting automatically.
