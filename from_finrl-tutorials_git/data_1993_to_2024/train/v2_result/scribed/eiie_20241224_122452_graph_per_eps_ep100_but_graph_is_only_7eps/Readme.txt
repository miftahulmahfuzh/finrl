this experiment shows the new 'graph_per_eps' that redraw the graph for train, dev, test at the end of single episode step, not AFTER the episode loop
so here, the parameter 'episodes' in model_conf.json is set to 100, but we can already see the graph.png during the training (the graph is only at 7 episodes)
this is much better that the previous code that only draw the final graph at the end of episode loop in algorithm.py
so now, if the training crashed midrun, we can still see the plot of portfolio up until the last crashed episode