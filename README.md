# Combine the [Leffingwell and Goodscents](https://github.com/pyrfume/pyrfume-data) (molecule, odor) data sets.

Running `julia "process_scent_data.jl"` downloads, processes, and combines the two datasets.
It is also a [Pluto](https://plutojl.org/) notebook.

`"process_scent_data.html"` shows the notebook output.

`"pyrfume.csv"` contains the resultant data set (structures and labels).

`"odor_key.json"` contains a dictionary that converts bidirectionally between column number and odor label.
