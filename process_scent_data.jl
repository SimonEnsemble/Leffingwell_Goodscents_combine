### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 99004b2e-36f7-11ed-28ae-f3f75c823964
begin
	import Pkg; Pkg.activate()#"."; io=devnull)
	using CSV, DataFrames, JSON
	using CairoMakie, Colors, PlutoTest, PlutoUI
	TableOfContents(; title="Odor Data Processing", depth=3)
end

# ╔═╡ e3e38b40-e41d-4afc-9054-a253287a82bb
update_theme!(;
		resolution=(500, 400), 
		font="Noto Sans", 
		linewidth=3, 
		backgroundcolor="#FAF7F2", 
		Axis=(;
			xgridstyle=:dash, 
			ygridstyle=:dash, 
			xtickalign=1, 
			ytickalign=1, 
			titlefont="Noto Sans"
		),
		palette=(;
			color=[
				"#34495e", 
				"#3498db", 
				"#2ecc71", 
				"#f1c40f", 
				"#e74c3c", 
				"#9b59b6", 
				"#1abc9c", 
				"#f39c12",
				"#d35400"
			],
			marker=[
				:circle,
				:utriangle, 
				:cross, 
				:rect, 
				:diamond, 
				:dtriangle, 
				:pentagon, 
				:xcross
			]
		)
	)

# ╔═╡ 6946a79a-8141-4028-b3e9-db109437a407
md"""
# Objective
"""

# ╔═╡ b0a910b0-2a75-4db5-8b8d-9bbb2b46dc64
md"""
Pre-process, combine, and summarize the odorant perception data of _Goodscents_ and _Leffingwell_, obtained from [_The Pyrfume Project's_](https://pyrfume.org/) data [repository](https://github.com/pyrfume/pyrfume-data/). 
"""

# ╔═╡ adedc0fc-5651-47c1-a687-bc6952cdd894
md"""
We combine these data mimicking the procedure described in:

!!! citation ""	
	[\"Machine Learning for Scent: Learning Generalizable Perceptual Representations of Small Molecules\"](https://arxiv.org/abs/1910.10685).

This study treated the machine learning task of odorant perception as a multi-class classification problem: a molecule is input to the machine learning model, and the output is the predicted odor descriptor for the molecule. Read more at the [Google AI blog](https://ai.googleblog.com/2019/10/learning-to-smell-using-deep-learning.html). A more recent follow-up olfactory study was published in _Science_ [here](https://www.science.org/doi/10.1126/science.ade4401).
"""

# ╔═╡ 7fc6a021-b62b-4293-88a6-d564468deebd
md"""
# Pre-Process Sources
"""

# ╔═╡ 19b9aae7-02dc-4f73-81b6-e48f22421078
md"""
## Description
"""

# ╔═╡ 5c2147bb-7f80-421a-8e03-908d3f99e3e6
md"""
### Odors
"""

# ╔═╡ f5f1a79a-2422-4c37-9e7a-40e5bdbac693
md"""
Each instance in the tabular data represents the outcome of an experiment where a human is given a sample of a pure compound and asked to describe what it smells like.
"""

# ╔═╡ 90169515-5113-4563-b36c-b7e9c2aefc81
md"""
### Molecules
"""

# ╔═╡ dc2863e3-2cb1-40c9-ac64-de422553c820
md"""
The [simplified molecular-input line-entry system (SMILES)](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) is used to specify the molecular structure of the substance provided in the olfaction survey.
"""

# ╔═╡ d86cea70-e801-4a1b-aacb-80ffb03781ba
md"""
## GitHub Commit
"""

# ╔═╡ 80ba1d37-955a-4ae2-82ef-50455a70985c
md"""
We use the data from the following commit, which was the most recent as of mid-February 2024.  `pyrfume` has an API for downloading the most recent commit, but not for a specific commit, so we enforce data stability this way.
"""

# ╔═╡ 0ee9fcde-d4af-4b86-9252-df991e60268b
pyrfume_commit = "0e9652c4fde822ad4009d175feb4cfeb7e80d3f7";

# ╔═╡ 9408afd6-e0e4-4f09-8f9e-b2ad40960f81
md"""
## Downloading Raw Data
"""

# ╔═╡ 7cf09702-6e16-432e-9f63-73ca7ffba8c4
md"""
Each raw data file will be downloaded from GitHub and loaded into a `DataFrame`.
"""

# ╔═╡ 1d3a8bb4-1e6c-4961-b88e-ddd48d8ec020
"""
download a specific commit of a file from the pyrfume GitHub data repository and return the CSV data as a `DataFrame`
"""
function get_raw_data(file; commit=pyrfume_commit)
	@assert endswith(file, ".csv")
	url = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/$commit/$file"
	filepath = download(url)
	df = CSV.read(filepath, DataFrame)
	return df
end

# ╔═╡ 328c17c1-bac3-4c61-80fd-233bf0f84cc0
md"""## Read and Process Leffingwell
"""

# ╔═╡ c30f3299-767a-46f6-b9e1-ee4585e99ff7
md"""
### Behaviors
"""

# ╔═╡ 22146540-9de7-4040-a6ac-cbf816f7b78e
md"""
`behaviors_sparse.csv` lists the olfactory perception labels of the molecules in the Leffingwell data set, which we load as `leffingwell_behaviors`:
"""

# ╔═╡ dc7bd01b-17d7-4276-986e-e273d6072430
leffingwell_behaviors = get_raw_data("leffingwell/behavior_sparse.csv")

# ╔═╡ 9e046a92-90f5-4b6a-8fe2-5b741e761f15
md"""
### Stimuli
"""

# ╔═╡ b0350dd9-0d90-4cc5-8b48-dfb8ce7dfda0
md"""
`stimuli.csv` lists various identifiers for each chemical stimulus, along with their molecular weights and SMILES representations.
"""

# ╔═╡ 00f67da6-4d8f-46cf-bdf1-4baba2b33253
leffingwell_stimuli = get_raw_data("leffingwell/stimuli.csv")

# ╔═╡ 8277ab12-b194-42e1-9802-46feff993227
md"""
### Join
"""

# ╔═╡ 463188aa-615c-4426-b7de-50cd816716f7
md"""
Join `leffingwell_stimuli` and `leffingwell_behaviors` using each molecule's unique `"Stimulus"`, drop all columns except `"IsomericSMILES"` and `"Labels"`, and rename these to `"molecule"` and `"odor"`, and extract the individual labels from the raw strings.
"""

# ╔═╡ f7d020c8-50be-4329-8551-907edf81d605
"""
transform a column of values by removing square brackets and commas and splitting on single quotes, followed by discarding artifact strings
"""
function transform_leffingwell(col)
	# remove invalid characters
	replacement_pairs = ["$x" => "" for x in "[,]"]
	col = [replace(row, replacement_pairs...) for row in col]
	# split on single quotes
	col = split.(col, '\''; keepempty=false)
	# remove the single-space strings that are left behind
	return [filter(≠(" "), s) for s in col]
end

# ╔═╡ 19613f27-7358-417e-b77d-15d734a88e42
leffingwell = let
	df = outerjoin(leffingwell_behaviors, leffingwell_stimuli, on="Stimulus")
	select!(df, ["IsomericSMILES", "Labels"])
	rename!(df, "Labels" => "leffingwell_odor", "IsomericSMILES"=>"molecule")
	transform!(df, "leffingwell_odor" => transform_leffingwell; renamecols=false)
end

# ╔═╡ a71a4c95-b3e7-49c7-b729-236869118729
md"""
Check that all molecules have at least one odor label.
"""

# ╔═╡ 378f3ff8-e693-4852-886b-b192eed33f67
@assert all(length.(leffingwell.leffingwell_odor) .> 0)

# ╔═╡ 8f5e6978-9016-4bff-ae41-033f5be4219a
md"""
Check for SMILES uniqueness.
"""

# ╔═╡ bc94bbbe-c2e4-46bf-804c-e51256b2373d
@assert length(unique(leffingwell[:, "molecule"])) == nrow(leffingwell)

# ╔═╡ 5ba1bb77-e905-4508-aa72-c8e5e168b616
md"""
## Read and Process Goodscents 
"""

# ╔═╡ ff0b9207-68f7-4218-85b5-5995de34ab61
md"""
### Behaviors
"""

# ╔═╡ 7ee15b57-15f1-4b64-ac09-53e9d0dbb550
md"""
Load `behaviors.csv` from Goodscents. Perception labels are stored in the `"Descriptors"` column, and the identity of the molecule is indicated by the `"Stimulus"` column.
"""

# ╔═╡ 07ce7883-f61f-4113-99ba-2195b4e41dfa
goodscents_behaviors = get_raw_data("goodscents/behavior.csv")

# ╔═╡ d0979277-3d2d-431e-af4c-d87a3ba7cb03
md"""
### Stimuli
"""

# ╔═╡ 2b826866-62dc-4f53-bccc-658602f624a7
md"""
Read `stimuli.csv`.
"""

# ╔═╡ f7d72939-f78d-476a-978a-b48fe6a51cec
goodscents_stimuli = get_raw_data("goodscents/stimuli.csv")

# ╔═╡ 5d229608-2d37-430a-97e1-cd95f0e44058
md"""
### Molecules
"""

# ╔═╡ 3cee2486-1904-43a6-933a-303d933267a6
md"""
Read `molecules.csv`.
"""

# ╔═╡ bd1bdb6a-51e8-43f3-85d5-cbf757686cc8
goodscents_molecules = get_raw_data("goodscents/molecules.csv")

# ╔═╡ 1542471a-0877-4a18-9f48-6f4d96b12e2a
md"""
### Join
"""

# ╔═╡ 25b4bda4-f2f6-436f-90ec-01c0594118b5
md"""
Join the three tables so as to link together (i) the molecule described by its SMILES string with (ii) its odor perception labels.  Rename `"IsomericSMILES"` to `"molecule"`, rename `"Descriptors"` to `"odor"`, and drop all columns except `"molecule"` and `"odor"`.  Finally, drop all rows with `missing` data, and group by molecule to find duplicate SMILES.
"""

# ╔═╡ dfd599fa-7991-48cc-b7e0-a0bb0bf9fe11
begin
	# join tables on Stimulus
	_goodscents = outerjoin(goodscents_behaviors, goodscents_stimuli; on="Stimulus")
	# join with molecules on CID
	_goodscents = innerjoin(_goodscents, goodscents_molecules; on="CID")
	# rename and select columns
	rename!(_goodscents, "IsomericSMILES" => "molecule")
	rename!(_goodscents, "Descriptors" => "goodscents_odor")
	select!(_goodscents, ["molecule", "goodscents_odor"])
	# drop rows with missing data
	dropmissing!(_goodscents)
	# # extract the labels from the raw strings
	transform!(
		_goodscents, 
		"goodscents_odor" => col -> [String.(x) for x in split.(col, ";")]; 
		renamecols=false
	)
end

# ╔═╡ 600197f1-a9d5-478b-8fa0-6deac3d25b57
goodscents_gdf = groupby(_goodscents, :molecule)

# ╔═╡ 2e9eb9ec-7d3c-4042-8d52-1dbc306c7330
md"""
Groups representing duplicate and triplicate SMILES:
"""

# ╔═╡ 0f90911c-d106-4e6a-8260-4022085ff5b7
ids_duplicates = findall([nrow(x) for x in goodscents_gdf] .> 1)

# ╔═╡ 4578e868-c562-4e55-b107-5f57f9278f67
length(ids_duplicates)

# ╔═╡ 74678b17-ea44-4f3d-bedc-ba13a7c3d8c7
duplicate_molecule = goodscents_gdf[ids_duplicates[1]][1, "molecule"]

# ╔═╡ a7e2b7c9-ebfb-45d0-95ac-9c41d7dc4fed
goodscents_gdf[ids_duplicates[1]]

# ╔═╡ 54bdbcdc-dff8-4771-b1d7-a89c0f4b164c
md"""
Merge rows by concatenating odor lists

| molecule | goodscents_odor |
| --- | ----------- |
| \"C(=O)O\" | \"acetic vinegar pungent\" |
| \"C(=O)O\" | \"acetic fermented sharp fruity\" |

--->

| molecule | goodscents_odor |
| --- | ----------- |
| \"C(=O)O\" | \"acetic vinegar pungent acetic fermented sharp fruity\" |
"""

# ╔═╡ d962f970-e62c-4a05-8219-2c81a54bdaba
"""
combine a column from a `GroupedDataFrame` into a single odor list per group
"""
function combine_goodscents(gcol)
	# gcol is the odor column from a single group
	if length(gcol) == 1
		# the group is already formatted properly
		return gcol
	else
		# the group needs to be merged
		new_col = reduce(union, gcol)
		return [new_col]
	end
end

# ╔═╡ 9ad586da-ccbc-46e8-8e93-7f59ffb3bfd3
goodscents = combine(
	goodscents_gdf, 
	"goodscents_odor" => combine_goodscents; 
	renamecols=false
)

# ╔═╡ 4eb6f6a7-a130-40d8-b69f-0f1d8517b72f
filter(row -> row["molecule"] == duplicate_molecule, goodscents)

# ╔═╡ fa0df4f6-51f8-47cf-8433-09f427833505
@test length(goodscents_gdf) == nrow(goodscents)

# ╔═╡ cdac62a7-5ed3-493b-949d-4796b9ead3dd
md"""
# Join Data Sets
"""

# ╔═╡ e9f70bdb-a67e-4174-b5dd-cc7df6fd3964
md"""
## Merge
"""

# ╔═╡ 59f4d262-c916-4f62-a1c1-11fda7360027
md"""
Join `leffingwell` and `goodscents` on `"molecule"`, replace each occurance of `missing` with an empty `Vector`, and then take the union of the labels from the two data sets.
"""

# ╔═╡ e03e0656-9198-4419-ac8b-97ef98f37c3f
merged = let
	df = outerjoin(goodscents, leffingwell; makeunique=true, on="molecule")
	df = coalesce.(df, [[]])
	transform!(
		df, 
		["goodscents_odor", "leffingwell_odor"] => 
			((col1, col2) -> col1 .∪ col2) => 
			"odor"
	)
	select!(df, [:molecule, :odor])
end

# ╔═╡ 55a1ed69-003a-4702-8226-80cc050a0912
md"to check, find some molecule in common between the two data sets and make sure they merged correctly."

# ╔═╡ 80c3eaa2-dfa2-4822-9dc6-002dd3fc23c9
begin
	common_molecule = pop!(intersect(
		Set(goodscents[:, "molecule"]),
		Set(leffingwell[:, "molecule"]),
	))
	
	all_odors = vcat(
		filter(
			row -> row["molecule"] == common_molecule, goodscents
		)[1, "goodscents_odor"],
		filter(
			row -> row["molecule"] == common_molecule, leffingwell
		)[1, "leffingwell_odor"]
	)
	
	@test filter(
		row -> row["molecule"] == common_molecule, merged
	)[1, "odor"] == all_odors
end

# ╔═╡ ebd5497a-799b-4064-b8ac-0365147fb4f8
md"""
## Odor label replacements
we'll store them here then replace at the end.
"""

# ╔═╡ a5ac81f7-2000-40d9-8969-b599fb18ae0d
odor_label_replacements = Dict{String, String}()

# ╔═╡ c14edce9-c764-4938-b4d3-cdbba49433d1
prelim_odors = String.(reduce(union, merged.odor))

# ╔═╡ f732f409-283a-46f3-bcc1-9b8913f37fda
md"""
#### "ABA" Labels
"""

# ╔═╡ 057fff1b-e302-4dcb-9817-199cc2d0bca7
md"""
Some labels are composites of the form "A B A", like "cherry maraschino cherry".
We convert these to simply "A", as the "B" word is mostly meaningless in isolation:
"""

# ╔═╡ 691057f2-0cbf-482b-8b4c-58601d8f7bcf
function is_aba(str)
	tokens = split(str)
	return length(tokens) == 3 && tokens[1] == tokens[3]
end

# ╔═╡ ae23e1fd-e5b6-4829-8fb6-32aa258719e1
aba_labels = filter(is_aba, prelim_odors)

# ╔═╡ df1e2872-3661-419c-94cb-326899f94efd
for o in aba_labels
	odor_label_replacements[o] = split(o)[1]
	println(o, " => ", odor_label_replacements[o])
end

# ╔═╡ 0a729562-155c-4911-9c30-40f79c41ecab
md"""
### Cheesy Cheese
"""

# ╔═╡ 18df3697-ce68-4697-a27b-1bb74952ec47
md"""
Some labels are of the particular format "cheesy X cheese", where "X" is a particular cheese variety, e.g. parmesan.
These are each reduced simply to "cheesy"
"""

# ╔═╡ f709afff-cede-4af6-bcb9-1df480c3c7e0
function is_cheesy_cheese(str)
	tokens = split(str)
	return length(tokens) == 3 && tokens[1] == "cheesy" && tokens[3] == "cheese"
end

# ╔═╡ 22907771-f6d7-47b2-8aaf-ee9fbfc96f9c
cheesy_corrections = filter(o -> is_cheesy_cheese(o), prelim_odors)

# ╔═╡ 775e1f89-12c7-4078-8097-3e906a8b1973
for o in cheesy_corrections
	odor_label_replacements[o] = "cheese"
	println(o, " => ", odor_label_replacements[o])
end

# ╔═╡ e2326792-be54-4a14-ab8d-04019010c372
md"""
#### Multi-Word Labels
"""

# ╔═╡ fd4f1370-5043-4fda-897e-9a62291cbcba
md"""
Some labels are multi-word phrases, like "lemon peel", where the second word contributes very little.
The second word of these phrases can be dropped.
"""

# ╔═╡ 4ff07081-dd7e-4646-9bcb-9f15b0c75d89
words_to_drop = [
	" skin", " peel", " rind", " leaf", " needle", " yolk", " root", " chip", " flesh", " seed", " fat", " juice", " butter"
];

# ╔═╡ 47a135c0-fe63-4a13-a567-f3cd428ab73e
function two_words_second_meaningless(str)
	for word in words_to_drop
		if occursin(word, str)
			return replace(str, word => "")
		end
	end
	return str
end

# ╔═╡ 6e06d55b-34c4-4937-ae89-e40555cc5bd6
for o in prelim_odors
	for word in words_to_drop
		if occursin(word, o)
			odor_label_replacements[o] = replace(o, word => "")
			println(o, " => ", odor_label_replacements[o])
		end
	end
end

# ╔═╡ 64176145-8bb6-4403-890c-a7a981e710fb
md"""
### Currant
"""

# ╔═╡ 1c2842db-d4ac-4e38-94cb-7c9638455c91
currant_forms = filter(o -> occursin("currant", o), prelim_odors)

# ╔═╡ e49adaa0-c250-4610-a2f1-baad018867df
"""
The odor of blackcurrant is encoded $(length(currant_forms)) different ways:
""" |> Markdown.parse

# ╔═╡ 740e23a3-d701-417d-a3b9-659f6a072a0b
md"""
We correct instances of the two alternate forms to `"currant"`.
"""

# ╔═╡ 09996b05-3094-4b51-8b54-e0856f36786a
for o in currant_forms
	odor_label_replacements[o] = "currant"
	println(o, " => ", odor_label_replacements[o])
end

# ╔═╡ e0986408-b153-47f6-89e3-8d18b454744f
@test false
# why does "currant bud currant bud" occur still?

# ╔═╡ b919dd0d-9bc4-46eb-bea2-d6e37da5aaef
md"""
### Noun/Adjective Pairs
"""

# ╔═╡ 0628a48a-e56c-404f-bcb9-c4ee77cd4423
md"""
Some labels are present both in noun and adjective form, e.g. "fish" and "fishy"
"""

# ╔═╡ ddd438af-9bed-489e-97b6-0e2545e05fc1
md"""
These should all be coerced to adjective form:
"""

# ╔═╡ 5809737f-edac-46e3-a5a8-941c8520efe2
function is_nouny(str)
	for x in prelim_odors
		if str == "$(x)y"
			return true
		end
	end
	return false
end

# ╔═╡ 8050b56a-5fd5-4cd4-b69f-b20c717270ae
nouny_odors = filter(o -> is_nouny(o), prelim_odors)

# ╔═╡ fa4b6f62-a849-4372-941d-1a140e467082
for o in nouny_odors
	odor_label_replacements[o] = o[1:end-1]
	println(o, " => ", odor_label_replacements[o])
end

# ╔═╡ b94b053c-0834-4854-961e-b78b4e6dfc9d
md"#### a few more manual replacements"

# ╔═╡ 7de3ca35-3678-4993-9699-658ba0a0493f
md"Concorde grape => grape"

# ╔═╡ 87bc0133-c5e0-46eb-8aa4-92348ace1389
odor_label_replacements["concorde grape"] = "grape"

# ╔═╡ bf5649b4-8943-4fc0-ba87-e4fc1b6575bf
odor_label_replacements["bread baked"] = "bread"

# ╔═╡ 1d1d3868-43f5-4531-97cf-79a089ab793b
md"#### finally, replace the odor labels"

# ╔═╡ ceecd769-357c-4bfe-aefa-c479ce6bab4e
odor_label_replacements

# ╔═╡ 06a5887c-73d5-41d4-a3a4-661ac6e362c9
transform!(
	merged,
	:odor => col -> [replace(row, odor_label_replacements...) for row in col] .|> unique;
	renamecols=false
)

# ╔═╡ 95513350-d56d-4d68-a97a-dc77b7bfd495
length(prelim_odors)

# ╔═╡ 171258ed-f732-44c2-8d8d-4a386ac8f5f2
md"""
## Final Processing
"""

# ╔═╡ f149b713-6e4c-462b-8cc3-ebc2d433d1f8
md"""
### Label Frequency
"""

# ╔═╡ 2020a24e-8dbe-4f2a-9aa3-5999391e7d9d
md"""
The remaining multi-word labels are sufficiently rare to excuse their exclusion.
In fact, all labels with fewer than 30 positive instances are excluded.
"""

# ╔═╡ 1ee63411-94a2-4535-a6bf-1ce62e23a6db
examples_per_anomalous_label = [
	anomaly => count(isequal(anomaly), reduce(vcat, merged.odor)) 
	for anomaly in filter(
		x -> occursin(" ", x), reduce(union, merged.odor)
	)
]

# ╔═╡ 514c4b55-f246-475d-b5bc-67349854e0b4
md"""
This yields molecules with no labels left.
These must be removed.
"""

# ╔═╡ 39a1c22a-4011-4b15-9604-fd8ebcfc3357
md"""
### Success!
"""

# ╔═╡ 29ea157b-a324-49a0-8412-03d03be9b6e7
md"""
## Bitvector Encoding
"""

# ╔═╡ 2462e47a-4f17-4723-87f8-324a2a570706
md"""
We need these data in bit-encoded format.
"""

# ╔═╡ a3e335ba-ad0a-4c59-865a-905bd8d4aaa0
md"some checks"

# ╔═╡ 7c889e04-bbee-490b-8cf8-fcc88fca3712
md"""
# Write to File
"""

# ╔═╡ 38810e1d-94d9-4a18-8346-919cc1dba734
md"""
## CSV
"""

# ╔═╡ 9302a9af-ce89-4d2a-a46b-573e3b4257a9
md"""
Structures and odor label bitvectors
"""

# ╔═╡ 4f4c5f24-ed9c-4576-8804-c11b320885f4
md"""
## JSON
"""

# ╔═╡ 7204fe2c-ae72-4a7f-8d89-692bba18517d
md"""
Odor label encoding/decoding key
"""

# ╔═╡ 667f1ff5-d22c-4b8c-b5a4-1148f6741202
md"""
# Analysis
"""

# ╔═╡ 905dc26a-fc2c-47a0-8569-a4b7a4541cfa
md"""
Goal: conduct a similar analysis of the olfactory data as in Fig. 3 [here](https://arxiv.org/pdf/1910.10685.pdf).
"""

# ╔═╡ b520bcf8-1aee-4a17-8e34-4ce97206bd7c
md"""
### Odor Labels per Molecule
"""

# ╔═╡ 28e4e0c5-858d-4d7c-b7d2-3771ab7affb1
md"""
List the number of unique olfactory perception labels on each molecule in the data.
"""

# ╔═╡ a11f06bb-9fcc-431d-9967-f8d26aa44bf2
analyzed_data = transform(
	label_freq_corrected2, 
	"odor" => (col -> map(row -> length(row), col)) => "# odor labels"
)

# ╔═╡ b562ad25-45e5-4fc9-8d74-27b9727e4766
md"List the number of molecules with a given number of odor labels.

| # odor labels | # molecules |
| ---- | ---- |
| 1 | 703 | 
| 2 | 665 |
| ... | ... |

"

# ╔═╡ 0233d3e0-c934-48c2-be82-8e041227aec5
odor_label_counts = combine(
	groupby(analyzed_data, "# odor labels"), nrow => "# molecules"
)

# ╔═╡ ec3b2463-4284-41bb-9c60-63ba8a7b6705
begin
	fig2 = Figure(; size=(660, 400))
	ax2  = Axis(fig2[1, 1],
		xlabel="# odor labels on a molecule",
		ylabel="# molecules",
		xticks=1:25
	)
	barplot!(odor_label_counts[:, "# odor labels"], 
		    odor_label_counts[:, "# molecules"])
	# ylims!(-1, nrow(odors)+1)
	fig2
end

# ╔═╡ 104cf55d-538c-4b5c-a6eb-47c91f20aa6f
md"""
### Label Prevalence
"""

# ╔═╡ 25a5c232-a0f6-4a1b-8f91-3cee5d97b3db
begin
	expanded_data = DataFrame(molecule=String[], odor=String[])
	for row in eachrow(label_freq_corrected2)
		for odor in row["odor"]
			push!(expanded_data, [row["molecule"], odor])
		end
	end
	expanded_data
end

# ╔═╡ 5765952b-8b4e-4a4f-bb38-75a49a5f15ce
@test !("" in odors)

# ╔═╡ 6318dcf4-4478-4398-bbfc-5c663c078de9
@test length(filter(o -> o in keys(odor_label_replacements), odors)) == 0

# ╔═╡ 6c547994-0566-4480-b23d-fbdc437431b0
length(odors)

# ╔═╡ 600de752-8927-493f-b58e-605773bb6c4c
counts_per_label = Dict(
	o => count(
		isequal(o), reduce(vcat, merged.odor)
	) for o in odors
)

# ╔═╡ d0ef14c9-391c-4af8-b915-d86c8aefd002
function is_over_threshold(o)
	return counts_per_label[o] ≥ 30
end

# ╔═╡ 47cb7741-01da-4bd4-8c05-c818e6c87f6b
trimmed_merged = transform(
	merged,
	:odor => col -> [filter(≠(""), [is_over_threshold(o) ? o : "" for o in row]) for row in col];
	renamecols=false
)

# ╔═╡ 83d1567c-b101-4f59-a658-1ff20e8d8b0f
length.(trimmed_merged.odor) |> minimum

# ╔═╡ 394a567a-8e4e-4c6e-8f22-69e31daca417
filter!(row -> length(row.odor) > 0, trimmed_merged)

# ╔═╡ 9deb1a41-1248-47da-aacb-c129863f8db7
"""
Every molecule now has at least $(length.(trimmed_merged.odor) |> minimum) label...
""" |> Markdown.parse

# ╔═╡ 20380a67-3b2d-4063-a491-5bfe68706ee8
@test length.(trimmed_merged.odor) |> minimum == 1

# ╔═╡ 3f8057f7-3391-4361-9ec5-7f739c66649c
md"""
...and there are $(length(reduce(union, trimmed_merged.odor))) labels that are sufficiently frequent.
"""

# ╔═╡ 5c7647f0-ed46-4d14-90d7-c95584d45cc4
length(reduce(union, trimmed_merged.odor))

# ╔═╡ c43ae3db-a34c-4aab-815e-f6820b44e558
new_counts_per_label = Dict(
	o => count(
		isequal(o), reduce(vcat, trimmed_merged.odor)
	) for o in reduce(union, trimmed_merged.odor)
)

# ╔═╡ f1403f59-3a34-4be8-a5fc-1f04513b80db
@test all(values(new_counts_per_label) .>= 30)

# ╔═╡ 1a0c8ab5-6181-48a0-8a91-02a9bd2c75a4
label_to_idx = let
	label_vec = reduce(union, trimmed_merged.odor) 
	Dict(label_vec .=> eachindex(label_vec))
end

# ╔═╡ fe2a4aab-85ce-4c2b-b042-3bdf5bf8fba2
idx_to_label = Dict(v => k for (k, v) in label_to_idx)

# ╔═╡ fda94ce1-4069-4320-804a-d1f83d9f1073
function odor_list_to_vector_encoding(odor_list)
	vector = zeros(Int, length(label_to_idx))
	for odor in odor_list
		vector[label_to_idx[odor]] = 1
	end
	return vector
end

# ╔═╡ 35c4865e-24a2-4196-90e1-5c6c6a770a04
open("odor_key.json"; write=true) do f
	json = JSON.json(merge(idx_to_label, label_to_idx))
	write(f, json)
end;

# ╔═╡ 8584efe5-be9e-42ba-973d-4634bf6ec1bb
data = let
	# df = trimmed_merged
	data = transform(
		trimmed_merged,
		:odor => col -> [odor_list_to_vector_encoding(row) for row in col];
		renamecols=false
	)
	mat = reduce(hcat, data.odor)' |> Matrix
	cols = [idx_to_label[i] => copy(col) for (i, col) in enumerate(eachcol(mat))]
	DataFrame("molecule" => trimmed_merged.molecule, cols...)
end

# ╔═╡ 961f8c2c-cf31-47cc-ba96-14ded08c7507
CSV.write("pyrfume.csv", data);

# ╔═╡ 28ec1019-f51e-4dd9-a5ca-cfd0a095fcea
begin
	fig = Figure(; resolution=(500, 10000))
	ax  = Axis(fig[1, 1], 
		xlabel="# molecules", 
		ylabel="odor", 
		title="odor prevalence",
		yticks=(1:nrow(odors), odors[:, "odor"])
	)
	xlims!(0, nothing)
	barplot!(1:nrow(odors), odors[:, "# molecules"], direction=:x)
	ylims!(0.0, nrow(odors)+0.5)
	fig
end

# ╔═╡ 3d1f7cfa-ae9b-4ed1-9261-595fee65a974
# ╠═╡ disabled = true
#=╠═╡
odors = sort!(
	combine(groupby(expanded_data, "odor"), nrow => "# molecules"), 
	"# molecules";
	rev=true
)
  ╠═╡ =#

# ╔═╡ 07d2caca-67da-4cd6-a329-3d241993f3ac
odors = String.(reduce(union, merged[:, "odor"])) # final odor list

# ╔═╡ Cell order:
# ╠═99004b2e-36f7-11ed-28ae-f3f75c823964
# ╟─e3e38b40-e41d-4afc-9054-a253287a82bb
# ╟─6946a79a-8141-4028-b3e9-db109437a407
# ╟─b0a910b0-2a75-4db5-8b8d-9bbb2b46dc64
# ╟─adedc0fc-5651-47c1-a687-bc6952cdd894
# ╟─7fc6a021-b62b-4293-88a6-d564468deebd
# ╟─19b9aae7-02dc-4f73-81b6-e48f22421078
# ╟─5c2147bb-7f80-421a-8e03-908d3f99e3e6
# ╟─f5f1a79a-2422-4c37-9e7a-40e5bdbac693
# ╟─90169515-5113-4563-b36c-b7e9c2aefc81
# ╟─dc2863e3-2cb1-40c9-ac64-de422553c820
# ╟─d86cea70-e801-4a1b-aacb-80ffb03781ba
# ╟─80ba1d37-955a-4ae2-82ef-50455a70985c
# ╠═0ee9fcde-d4af-4b86-9252-df991e60268b
# ╟─9408afd6-e0e4-4f09-8f9e-b2ad40960f81
# ╟─7cf09702-6e16-432e-9f63-73ca7ffba8c4
# ╠═1d3a8bb4-1e6c-4961-b88e-ddd48d8ec020
# ╟─328c17c1-bac3-4c61-80fd-233bf0f84cc0
# ╟─c30f3299-767a-46f6-b9e1-ee4585e99ff7
# ╟─22146540-9de7-4040-a6ac-cbf816f7b78e
# ╠═dc7bd01b-17d7-4276-986e-e273d6072430
# ╟─9e046a92-90f5-4b6a-8fe2-5b741e761f15
# ╟─b0350dd9-0d90-4cc5-8b48-dfb8ce7dfda0
# ╠═00f67da6-4d8f-46cf-bdf1-4baba2b33253
# ╟─8277ab12-b194-42e1-9802-46feff993227
# ╟─463188aa-615c-4426-b7de-50cd816716f7
# ╠═f7d020c8-50be-4329-8551-907edf81d605
# ╠═19613f27-7358-417e-b77d-15d734a88e42
# ╟─a71a4c95-b3e7-49c7-b729-236869118729
# ╠═378f3ff8-e693-4852-886b-b192eed33f67
# ╟─8f5e6978-9016-4bff-ae41-033f5be4219a
# ╠═bc94bbbe-c2e4-46bf-804c-e51256b2373d
# ╟─5ba1bb77-e905-4508-aa72-c8e5e168b616
# ╟─ff0b9207-68f7-4218-85b5-5995de34ab61
# ╟─7ee15b57-15f1-4b64-ac09-53e9d0dbb550
# ╠═07ce7883-f61f-4113-99ba-2195b4e41dfa
# ╟─d0979277-3d2d-431e-af4c-d87a3ba7cb03
# ╟─2b826866-62dc-4f53-bccc-658602f624a7
# ╠═f7d72939-f78d-476a-978a-b48fe6a51cec
# ╟─5d229608-2d37-430a-97e1-cd95f0e44058
# ╟─3cee2486-1904-43a6-933a-303d933267a6
# ╠═bd1bdb6a-51e8-43f3-85d5-cbf757686cc8
# ╟─1542471a-0877-4a18-9f48-6f4d96b12e2a
# ╟─25b4bda4-f2f6-436f-90ec-01c0594118b5
# ╠═dfd599fa-7991-48cc-b7e0-a0bb0bf9fe11
# ╠═600197f1-a9d5-478b-8fa0-6deac3d25b57
# ╟─2e9eb9ec-7d3c-4042-8d52-1dbc306c7330
# ╠═0f90911c-d106-4e6a-8260-4022085ff5b7
# ╠═4578e868-c562-4e55-b107-5f57f9278f67
# ╠═74678b17-ea44-4f3d-bedc-ba13a7c3d8c7
# ╠═a7e2b7c9-ebfb-45d0-95ac-9c41d7dc4fed
# ╟─54bdbcdc-dff8-4771-b1d7-a89c0f4b164c
# ╠═d962f970-e62c-4a05-8219-2c81a54bdaba
# ╠═9ad586da-ccbc-46e8-8e93-7f59ffb3bfd3
# ╠═4eb6f6a7-a130-40d8-b69f-0f1d8517b72f
# ╠═fa0df4f6-51f8-47cf-8433-09f427833505
# ╟─cdac62a7-5ed3-493b-949d-4796b9ead3dd
# ╟─e9f70bdb-a67e-4174-b5dd-cc7df6fd3964
# ╟─59f4d262-c916-4f62-a1c1-11fda7360027
# ╠═e03e0656-9198-4419-ac8b-97ef98f37c3f
# ╟─55a1ed69-003a-4702-8226-80cc050a0912
# ╠═80c3eaa2-dfa2-4822-9dc6-002dd3fc23c9
# ╟─ebd5497a-799b-4064-b8ac-0365147fb4f8
# ╠═a5ac81f7-2000-40d9-8969-b599fb18ae0d
# ╠═c14edce9-c764-4938-b4d3-cdbba49433d1
# ╟─f732f409-283a-46f3-bcc1-9b8913f37fda
# ╟─057fff1b-e302-4dcb-9817-199cc2d0bca7
# ╠═691057f2-0cbf-482b-8b4c-58601d8f7bcf
# ╠═ae23e1fd-e5b6-4829-8fb6-32aa258719e1
# ╠═df1e2872-3661-419c-94cb-326899f94efd
# ╟─0a729562-155c-4911-9c30-40f79c41ecab
# ╟─18df3697-ce68-4697-a27b-1bb74952ec47
# ╠═f709afff-cede-4af6-bcb9-1df480c3c7e0
# ╠═22907771-f6d7-47b2-8aaf-ee9fbfc96f9c
# ╠═775e1f89-12c7-4078-8097-3e906a8b1973
# ╟─e2326792-be54-4a14-ab8d-04019010c372
# ╟─fd4f1370-5043-4fda-897e-9a62291cbcba
# ╠═4ff07081-dd7e-4646-9bcb-9f15b0c75d89
# ╠═47a135c0-fe63-4a13-a567-f3cd428ab73e
# ╠═6e06d55b-34c4-4937-ae89-e40555cc5bd6
# ╟─64176145-8bb6-4403-890c-a7a981e710fb
# ╟─e49adaa0-c250-4610-a2f1-baad018867df
# ╠═1c2842db-d4ac-4e38-94cb-7c9638455c91
# ╟─740e23a3-d701-417d-a3b9-659f6a072a0b
# ╠═09996b05-3094-4b51-8b54-e0856f36786a
# ╠═e0986408-b153-47f6-89e3-8d18b454744f
# ╟─b919dd0d-9bc4-46eb-bea2-d6e37da5aaef
# ╟─0628a48a-e56c-404f-bcb9-c4ee77cd4423
# ╟─ddd438af-9bed-489e-97b6-0e2545e05fc1
# ╠═5809737f-edac-46e3-a5a8-941c8520efe2
# ╠═8050b56a-5fd5-4cd4-b69f-b20c717270ae
# ╠═fa4b6f62-a849-4372-941d-1a140e467082
# ╟─b94b053c-0834-4854-961e-b78b4e6dfc9d
# ╟─7de3ca35-3678-4993-9699-658ba0a0493f
# ╠═87bc0133-c5e0-46eb-8aa4-92348ace1389
# ╠═bf5649b4-8943-4fc0-ba87-e4fc1b6575bf
# ╟─1d1d3868-43f5-4531-97cf-79a089ab793b
# ╠═ceecd769-357c-4bfe-aefa-c479ce6bab4e
# ╠═06a5887c-73d5-41d4-a3a4-661ac6e362c9
# ╠═07d2caca-67da-4cd6-a329-3d241993f3ac
# ╠═5765952b-8b4e-4a4f-bb38-75a49a5f15ce
# ╠═6318dcf4-4478-4398-bbfc-5c663c078de9
# ╠═6c547994-0566-4480-b23d-fbdc437431b0
# ╠═95513350-d56d-4d68-a97a-dc77b7bfd495
# ╟─171258ed-f732-44c2-8d8d-4a386ac8f5f2
# ╟─f149b713-6e4c-462b-8cc3-ebc2d433d1f8
# ╟─2020a24e-8dbe-4f2a-9aa3-5999391e7d9d
# ╠═1ee63411-94a2-4535-a6bf-1ce62e23a6db
# ╠═600de752-8927-493f-b58e-605773bb6c4c
# ╠═d0ef14c9-391c-4af8-b915-d86c8aefd002
# ╠═47cb7741-01da-4bd4-8c05-c818e6c87f6b
# ╟─514c4b55-f246-475d-b5bc-67349854e0b4
# ╠═83d1567c-b101-4f59-a658-1ff20e8d8b0f
# ╠═394a567a-8e4e-4c6e-8f22-69e31daca417
# ╟─39a1c22a-4011-4b15-9604-fd8ebcfc3357
# ╟─9deb1a41-1248-47da-aacb-c129863f8db7
# ╠═20380a67-3b2d-4063-a491-5bfe68706ee8
# ╟─3f8057f7-3391-4361-9ec5-7f739c66649c
# ╠═5c7647f0-ed46-4d14-90d7-c95584d45cc4
# ╠═c43ae3db-a34c-4aab-815e-f6820b44e558
# ╠═f1403f59-3a34-4be8-a5fc-1f04513b80db
# ╟─29ea157b-a324-49a0-8412-03d03be9b6e7
# ╟─2462e47a-4f17-4723-87f8-324a2a570706
# ╠═1a0c8ab5-6181-48a0-8a91-02a9bd2c75a4
# ╠═fe2a4aab-85ce-4c2b-b042-3bdf5bf8fba2
# ╠═fda94ce1-4069-4320-804a-d1f83d9f1073
# ╠═8584efe5-be9e-42ba-973d-4634bf6ec1bb
# ╠═a3e335ba-ad0a-4c59-865a-905bd8d4aaa0
# ╟─7c889e04-bbee-490b-8cf8-fcc88fca3712
# ╟─38810e1d-94d9-4a18-8346-919cc1dba734
# ╟─9302a9af-ce89-4d2a-a46b-573e3b4257a9
# ╠═961f8c2c-cf31-47cc-ba96-14ded08c7507
# ╟─4f4c5f24-ed9c-4576-8804-c11b320885f4
# ╟─7204fe2c-ae72-4a7f-8d89-692bba18517d
# ╠═35c4865e-24a2-4196-90e1-5c6c6a770a04
# ╟─667f1ff5-d22c-4b8c-b5a4-1148f6741202
# ╟─905dc26a-fc2c-47a0-8569-a4b7a4541cfa
# ╟─b520bcf8-1aee-4a17-8e34-4ce97206bd7c
# ╟─28e4e0c5-858d-4d7c-b7d2-3771ab7affb1
# ╠═a11f06bb-9fcc-431d-9967-f8d26aa44bf2
# ╟─b562ad25-45e5-4fc9-8d74-27b9727e4766
# ╠═0233d3e0-c934-48c2-be82-8e041227aec5
# ╠═ec3b2463-4284-41bb-9c60-63ba8a7b6705
# ╟─104cf55d-538c-4b5c-a6eb-47c91f20aa6f
# ╠═25a5c232-a0f6-4a1b-8f91-3cee5d97b3db
# ╠═3d1f7cfa-ae9b-4ed1-9261-595fee65a974
# ╠═28ec1019-f51e-4dd9-a5ca-cfd0a095fcea
