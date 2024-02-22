### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 99004b2e-36f7-11ed-28ae-f3f75c823964
begin
	import Pkg; Pkg.activate("."; io=devnull)
	using CSV, DataFrames, LinearAlgebra, MolecularGraph
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
We combine these data following the procedure described in:

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

# ╔═╡ 8aee658c-cd64-4e43-b623-642ab0e66467
"""
transform a column of raw odor data by:
 - splitting on ';'
"""
function transform_goodscents(col)
	return [String.(x) for x in split.(col, ";")]
end

# ╔═╡ 600197f1-a9d5-478b-8fa0-6deac3d25b57
goodscents_gdf = let
	# join tables on Stimulus
	df = outerjoin(goodscents_behaviors, goodscents_stimuli; on="Stimulus")
	# join with molecules on CID
	df = innerjoin(df, goodscents_molecules; on="CID")
	# rename and select columns
	rename!(df, "IsomericSMILES" => "molecule")
	rename!(df, "Descriptors" => "goodscents_odor")
	select!(df, ["molecule", "goodscents_odor"])
	# drop rows with missing data
	dropmissing!(df)
	# extract the labels from the raw strings
	transform!(
		df, 
		"goodscents_odor" => transform_goodscents; 
		renamecols=false
	)
	# group by SMILES
	groupby(df, :molecule)
end

# ╔═╡ 2e9eb9ec-7d3c-4042-8d52-1dbc306c7330
md"""
Groups representing duplicate and triplicate SMILES:
"""

# ╔═╡ 0f90911c-d106-4e6a-8260-4022085ff5b7
findall([nrow(x) for x in goodscents_gdf] .> 1) |> length

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

# ╔═╡ ebd5497a-799b-4064-b8ac-0365147fb4f8
md"""
## Correct Anomalies
"""

# ╔═╡ 768e4669-fcb7-4f40-b050-4944be5e476e
md"""
### `"odorless"` Contradictions
"""

# ╔═╡ 71fef714-cab0-4329-8179-119b064943bd
odorless_idx = any.(isequal("odorless").(row) for row in merged.odor) |> findall

# ╔═╡ 52f5e5aa-93cb-4024-b61d-b912ff235802
"""
$(length(odorless_idx)) molecules have the `"odorless"` label:
""" |> Markdown.parse

# ╔═╡ c171078f-645f-42bb-83bd-563c4f2c8b8e
contradiction_idx = odorless_idx[length.(merged.odor[odorless_idx]) .> 1]

# ╔═╡ ea610639-2314-44f9-a640-dd59e7bd4694
"""
$(length(contradiction_idx)) of those co-occur with other labels!
""" |> Markdown.parse

# ╔═╡ 8ac56799-5b28-4b86-ace3-0aef423af0d7
merged[contradiction_idx, :]

# ╔═╡ 59ad90c7-a4a4-41e9-a28a-318f034a733b
md"""
This is a foreseeable problem when using data collected from subjective human experiences!  
For some people, a particular odor may be less perceptible.  
For the sake of logical consistency, we remove the `"odorless"` label from cases where it co-occurs with a contradictory label.
"""

# ╔═╡ e8b34946-8efc-4ee6-88f6-8eddbccffc4b
odorless_corrected = transform(
	merged,
	:odor => col -> [setdiff(row, ["odorless"]) for row in col];
	renamecols=false
)

# ╔═╡ f732f409-283a-46f3-bcc1-9b8913f37fda
md"""
### "ABA" Labels
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
aba_labels = filter(is_aba, reduce(union, odorless_corrected.odor))

# ╔═╡ 06cbe47f-0e80-4c95-996a-1d817c34f036
aba_corrected = transform(
	odorless_corrected,
	:odor => col -> [
		[is_aba(x) ? String.(split(x))[1] : x for x in row] for row in col
	];
	renamecols=false
)

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

# ╔═╡ 106d5c5d-edbb-4978-86ed-763eb2acb9f5
is_cheesy_cheese("cheesy cheddar cheese")

# ╔═╡ 30428033-9b5a-49c6-82da-2397b4d5af15
is_cheesy_cheese("watery coconut water")

# ╔═╡ 4bb4efa8-58ac-4bc6-9c01-6f7d1df9a511
cheesy_corrected = transform(
	aba_corrected,
	:odor => col -> [
		[is_cheesy_cheese(x) ? "cheesy" : x for x in row] for row in col
	];
	renamecols=false
)

# ╔═╡ b919dd0d-9bc4-46eb-bea2-d6e37da5aaef
md"""
### Noun/Adjective Pairs
"""

# ╔═╡ 0628a48a-e56c-404f-bcb9-c4ee77cd4423
md"""
Some labels are present both in noun and adjective form, e.g. "fish" and "fishy"
"""

# ╔═╡ 90e6c8b0-5535-4621-aab9-12ba450585e3
cheesy_corrected_labels = reduce(union, cheesy_corrected.odor);

# ╔═╡ 6c3fa2d5-5da1-4b8e-ad94-d47255141209
cheesy_corrected_labels[findall(startswith("fish"), cheesy_corrected_labels)]

# ╔═╡ ddd438af-9bed-489e-97b6-0e2545e05fc1
md"""
These should all be coerced to adjective form:
"""

# ╔═╡ 5809737f-edac-46e3-a5a8-941c8520efe2
function is_nouny(str)
	for x in cheesy_corrected_labels
		if str == "$(x)y"
			return true
		end
	end
	return false
end

# ╔═╡ 8050b56a-5fd5-4cd4-b69f-b20c717270ae
nouny_idx = is_nouny.(cheesy_corrected_labels) |> findall

# ╔═╡ fa4b6f62-a849-4372-941d-1a140e467082
nouny_pairs = [x[1:end-1] for x in cheesy_corrected_labels[nouny_idx]] .=> 
	cheesy_corrected_labels[nouny_idx]

# ╔═╡ e94a4f61-a01a-40e1-ae24-192a8cb41716
nouny_corrected = transform(
	cheesy_corrected,
	:odor => col -> [replace(row, nouny_pairs...) for row in col];
	renamecols=false
)

# ╔═╡ e2326792-be54-4a14-ab8d-04019010c372
md"""
### Multi-Word Labels
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
function peel_or_skin(str)
	for word in words_to_drop
		if occursin(word, str)
			return replace(str, word => "")
		end
	end
	return str
end

# ╔═╡ ddfe3e46-a152-4db1-93c2-b1ce7a864b59
peel_corrected = transform(
	nouny_corrected,
	:odor => col -> [peel_or_skin.(row) for row in col];
	renamecols=false
)

# ╔═╡ 64176145-8bb6-4403-890c-a7a981e710fb
md"""
### Currant
"""

# ╔═╡ 36627341-e5e9-44e7-973c-2c5db8dd6391
currant_forms = reduce(union, peel_corrected.odor)[
	findall(x -> occursin("currant", x), reduce(union, peel_corrected.odor))
]

# ╔═╡ e49adaa0-c250-4610-a2f1-baad018867df
"""
The odor of blackcurrant is encoded $(length(currant_forms)) different ways:
""" |> Markdown.parse

# ╔═╡ 740e23a3-d701-417d-a3b9-659f6a072a0b
md"""
We correct instances of the two alternate forms to `"currant"`.
"""

# ╔═╡ 09996b05-3094-4b51-8b54-e0856f36786a
currant_corrected = transform(
	peel_corrected,
	:odor => col -> [[replace(x, (currant_forms .=> ["currant"])...) for x in row] for row in col];
	renamecols=false
)

# ╔═╡ 171258ed-f732-44c2-8d8d-4a386ac8f5f2
md"""
## Final Data
"""

# ╔═╡ 2020a24e-8dbe-4f2a-9aa3-5999391e7d9d
md"""
The remaining multi-word labels are sufficiently rare to excuse their exclusion.
In fact, all labels with fewer than 30 positive instances are excluded.
"""

# ╔═╡ 1ee63411-94a2-4535-a6bf-1ce62e23a6db
examples_per_label = [
	anomaly => count(isequal(anomaly), reduce(vcat, currant_corrected.odor)) 
	for anomaly in filter(
		x -> occursin(" ", x), reduce(union, currant_corrected.odor)
	)
]

# ╔═╡ 47cb7741-01da-4bd4-8c05-c818e6c87f6b
data

# ╔═╡ 7c889e04-bbee-490b-8cf8-fcc88fca3712
md"# write to file"

# ╔═╡ 961f8c2c-cf31-47cc-ba96-14ded08c7507
CSV.write("pyrfume.csv", data);

# ╔═╡ 905dc26a-fc2c-47a0-8569-a4b7a4541cfa
md"## analysis of the joined data
the goal here is to conduct a similar analysis of the olfactory data as in Fig. 3 [here](https://arxiv.org/pdf/1910.10685.pdf).
"

# ╔═╡ 2a2910af-dcf2-4aa7-bfc3-32950d73cc9e
md"👃 like in Fig. 3a [here](https://arxiv.org/pdf/1910.10685.pdf) (yours will be a bit different), create a bar plot visualizing the number of molecules with a given number of olfactory labels on it. use your data frame `odor_label_counts` for this."

# ╔═╡ 9208b96f-963f-4fca-9a2e-3803160f11b1
md"
### odor labels per molecule
👃 append a new column to `data`, `\"# odor labels\"`, that lists the number of unique olfactory perception labels on each molecule in the data."

# ╔═╡ a11f06bb-9fcc-431d-9967-f8d26aa44bf2
# transform!(data, "odor" => (col -> map(row -> length(row), col)) => "# odor labels")

# ╔═╡ b562ad25-45e5-4fc9-8d74-27b9727e4766
md"👃 create a data frame `odor_label_counts` that lists the # of molecules with a given number of odor labels.

| # odor labels | # molecules |
| ---- | ---- |
| 1 | 703 | 
| 2 | 665 |
| ... | ... |

"

# ╔═╡ 0233d3e0-c934-48c2-be82-8e041227aec5
odor_label_counts = combine(groupby(data, "# odor labels"), nrow => "# molecules")

# ╔═╡ ec3b2463-4284-41bb-9c60-63ba8a7b6705
begin
	fig2 = Figure(resolution=(660, 400))
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

# ╔═╡ 53f58ff1-1f99-4205-9346-bf2f8ad4e74d
md"
### molecules that smell a certain way
👃 how many molecules in the data include a label \"eucalyptus\"?"

# ╔═╡ b1a04dc4-b281-413d-8cc0-3f704d1e6e3b
nrow(filter(row -> "eucalyptus" in row["odor"], data))

# ╔═╡ 82b4049e-5c34-43a4-85e4-fd95c1693989
md"👃 draw the molecular structure of the molecules that are labeled \"mint\" _and_ \"eucalyptus\".

!!! hint
	use `smilestomol` [here](https://mojaie.github.io/MolecularGraph.jl_notebook/molecular_graph_basics.jl.html).
"

# ╔═╡ 9e69ba26-4f61-4b54-94d8-986ed3d5a704
mint_and_eucalyptus = filter(
	row -> "mint" in row["odor"] && "eucalyptus" in row["odor"], 
	data
)

# ╔═╡ c8c928e8-f16c-4c6c-a807-5d64183e6d4d
smilestomol(mint_and_eucalyptus[1, "molecule"])

# ╔═╡ 832897f7-88fb-4389-b051-51adfec3db01
smilestomol(mint_and_eucalyptus[2, "molecule"])

# ╔═╡ 64bfc9fb-5866-426d-93ce-90521bf93356
smilestomol(mint_and_eucalyptus[3, "molecule"])

# ╔═╡ 47478ef0-a512-47d4-a171-74e073cab810
md"
### molecules per odor label
👃 we now wish to visualize the prevalence of each odor descriptor. create a bar plot that shows, for each unique odor label, how many molecules have that odor (according to the experts in the studies). 
* sort the bars according to odor prevalance
* make the bars go from left to right
* include the odor name on the y-axis
* include x- and y-axis labels (x-axis = # molecules, y-axis = odor)

in my view, this is a better way to present the (actually, MORE) information (than) in Fig. 3b [here](https://arxiv.org/pdf/1910.10685.pdf).

!!! warning
	this is very challenging. 

!!! hint
	to actually read the many odor labels, use `Figure(resolution=(500, 10000))` to make a long figure that you can scroll through.
"

# ╔═╡ 25a5c232-a0f6-4a1b-8f91-3cee5d97b3db
begin
	expanded_data = DataFrame(molecule=String[], odor=String[])
	for row in eachrow(data)
		for odor in row["odor"]
			push!(expanded_data, [row["molecule"], odor])
		end
	end
	expanded_data
end

# ╔═╡ 75e8312d-1614-4626-99d1-3e07fdec8bab
unique(expanded_data[:, "odor"])

# ╔═╡ 3d1f7cfa-ae9b-4ed1-9261-595fee65a974
odors = combine(groupby(expanded_data, "odor"), nrow => "# molecules")

# ╔═╡ fa50cac7-64f9-4439-9645-c6c0b4fa0b99
sort!(odors, "# molecules", rev=true)

# ╔═╡ 28ec1019-f51e-4dd9-a5ca-cfd0a095fcea
begin
	fig = Figure(resolution=(500, 10000))
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

# ╔═╡ 2045e894-396b-44d3-8dd8-ff0b88c598c2
md"
### molecules with certain chemistry
👃 how many of the odorants have a carbonyl group in their structure? this is expressed as the SMARTS pattern \"[CX3]=[OX1]\".

!!! hint
	see `has_substruct_match` [here](https://mojaie.github.io/MolecularGraph.jl_notebook/substructure_and_query.jl.html).
"

# ╔═╡ 86683b41-7b2a-471f-b817-bc0925c3770f
function has_carbonyl(smiles::String)
	mol = smilestomol(smiles)
	has_substruct_match(
			mol, smartstomol(raw"[CX3]=[OX1]"))
end

# ╔═╡ 493328f4-fcfd-49d4-84f7-6540ea6d2319
has_carbonyl(data[2, "molecule"])

# ╔═╡ 72101250-16ae-4c20-a7aa-e1479a41860f
nrow(
	filter(row -> has_carbonyl(row["molecule"]), data)
)

# ╔═╡ 58b6234b-a10e-46de-8031-ab65028d3c89
md"
### co-occurances of labels
the paper notes
> there is an extremely strong co-occurrence structure among odor descriptors that reflects a common-sense intuition of which odor descriptors are similar and dissimilar.

👃 (a strong co-occurance) among all molecules with \"apple\" as an olfactory label, what fraction of these also have the label `\"fruity\"?
"

# ╔═╡ 8bd8627b-9884-4efd-acf9-e1e57a52c833
n_apple = nrow(filter(row -> "apple" in row["odor"], data))

# ╔═╡ 06eabbc3-94af-4b76-9369-9e5df468cbbc
n_apple_and_fruity = nrow(filter(row -> ("apple" in row["odor"]) && ("fruity" in row["odor"]), data))

# ╔═╡ 494521db-50e9-48ee-913e-1ac3c70d3172
n_apple_and_fruity / n_apple

# ╔═╡ 380f9d36-fb68-490a-9021-16c265056a3e
md"👃 (a weak co-occurance) among all molecules with the label \"cabbage\", what fraction also have the label \"musk\"?"

# ╔═╡ c63f9137-574d-4c70-91ce-99fdc304bf02
n_cabbage = nrow(filter(row -> "cabbage" in row["odor"], data))

# ╔═╡ 1b3230db-bdda-4c71-9b33-31c779d50a7b
n_cabbage_and_musk = nrow(filter(row -> ("cabbage" in row["odor"]) && ("musk" in row["odor"]), data))

# ╔═╡ 7242a8c3-362c-4df9-b518-1a2a40ca2f39
n_cabbage_and_musk / n_cabbage

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
# ╠═8aee658c-cd64-4e43-b623-642ab0e66467
# ╠═600197f1-a9d5-478b-8fa0-6deac3d25b57
# ╟─2e9eb9ec-7d3c-4042-8d52-1dbc306c7330
# ╠═0f90911c-d106-4e6a-8260-4022085ff5b7
# ╟─54bdbcdc-dff8-4771-b1d7-a89c0f4b164c
# ╠═d962f970-e62c-4a05-8219-2c81a54bdaba
# ╠═9ad586da-ccbc-46e8-8e93-7f59ffb3bfd3
# ╠═fa0df4f6-51f8-47cf-8433-09f427833505
# ╟─cdac62a7-5ed3-493b-949d-4796b9ead3dd
# ╟─e9f70bdb-a67e-4174-b5dd-cc7df6fd3964
# ╟─59f4d262-c916-4f62-a1c1-11fda7360027
# ╠═e03e0656-9198-4419-ac8b-97ef98f37c3f
# ╟─ebd5497a-799b-4064-b8ac-0365147fb4f8
# ╟─768e4669-fcb7-4f40-b050-4944be5e476e
# ╟─52f5e5aa-93cb-4024-b61d-b912ff235802
# ╠═71fef714-cab0-4329-8179-119b064943bd
# ╟─ea610639-2314-44f9-a640-dd59e7bd4694
# ╠═c171078f-645f-42bb-83bd-563c4f2c8b8e
# ╠═8ac56799-5b28-4b86-ace3-0aef423af0d7
# ╟─59ad90c7-a4a4-41e9-a28a-318f034a733b
# ╠═e8b34946-8efc-4ee6-88f6-8eddbccffc4b
# ╟─f732f409-283a-46f3-bcc1-9b8913f37fda
# ╟─057fff1b-e302-4dcb-9817-199cc2d0bca7
# ╠═691057f2-0cbf-482b-8b4c-58601d8f7bcf
# ╠═ae23e1fd-e5b6-4829-8fb6-32aa258719e1
# ╠═06cbe47f-0e80-4c95-996a-1d817c34f036
# ╟─0a729562-155c-4911-9c30-40f79c41ecab
# ╟─18df3697-ce68-4697-a27b-1bb74952ec47
# ╠═f709afff-cede-4af6-bcb9-1df480c3c7e0
# ╠═106d5c5d-edbb-4978-86ed-763eb2acb9f5
# ╠═30428033-9b5a-49c6-82da-2397b4d5af15
# ╠═4bb4efa8-58ac-4bc6-9c01-6f7d1df9a511
# ╟─e2326792-be54-4a14-ab8d-04019010c372
# ╟─fd4f1370-5043-4fda-897e-9a62291cbcba
# ╠═4ff07081-dd7e-4646-9bcb-9f15b0c75d89
# ╠═47a135c0-fe63-4a13-a567-f3cd428ab73e
# ╠═ddfe3e46-a152-4db1-93c2-b1ce7a864b59
# ╟─b919dd0d-9bc4-46eb-bea2-d6e37da5aaef
# ╟─0628a48a-e56c-404f-bcb9-c4ee77cd4423
# ╠═90e6c8b0-5535-4621-aab9-12ba450585e3
# ╠═6c3fa2d5-5da1-4b8e-ad94-d47255141209
# ╟─ddd438af-9bed-489e-97b6-0e2545e05fc1
# ╠═5809737f-edac-46e3-a5a8-941c8520efe2
# ╠═8050b56a-5fd5-4cd4-b69f-b20c717270ae
# ╠═fa4b6f62-a849-4372-941d-1a140e467082
# ╠═e94a4f61-a01a-40e1-ae24-192a8cb41716
# ╟─64176145-8bb6-4403-890c-a7a981e710fb
# ╟─e49adaa0-c250-4610-a2f1-baad018867df
# ╠═36627341-e5e9-44e7-973c-2c5db8dd6391
# ╟─740e23a3-d701-417d-a3b9-659f6a072a0b
# ╠═09996b05-3094-4b51-8b54-e0856f36786a
# ╟─171258ed-f732-44c2-8d8d-4a386ac8f5f2
# ╟─2020a24e-8dbe-4f2a-9aa3-5999391e7d9d
# ╠═1ee63411-94a2-4535-a6bf-1ce62e23a6db
# ╠═47cb7741-01da-4bd4-8c05-c818e6c87f6b
# ╠═7c889e04-bbee-490b-8cf8-fcc88fca3712
# ╠═961f8c2c-cf31-47cc-ba96-14ded08c7507
# ╟─905dc26a-fc2c-47a0-8569-a4b7a4541cfa
# ╟─2a2910af-dcf2-4aa7-bfc3-32950d73cc9e
# ╟─9208b96f-963f-4fca-9a2e-3803160f11b1
# ╠═a11f06bb-9fcc-431d-9967-f8d26aa44bf2
# ╟─b562ad25-45e5-4fc9-8d74-27b9727e4766
# ╠═0233d3e0-c934-48c2-be82-8e041227aec5
# ╠═ec3b2463-4284-41bb-9c60-63ba8a7b6705
# ╟─53f58ff1-1f99-4205-9346-bf2f8ad4e74d
# ╠═b1a04dc4-b281-413d-8cc0-3f704d1e6e3b
# ╟─82b4049e-5c34-43a4-85e4-fd95c1693989
# ╠═9e69ba26-4f61-4b54-94d8-986ed3d5a704
# ╠═c8c928e8-f16c-4c6c-a807-5d64183e6d4d
# ╠═832897f7-88fb-4389-b051-51adfec3db01
# ╠═64bfc9fb-5866-426d-93ce-90521bf93356
# ╟─47478ef0-a512-47d4-a171-74e073cab810
# ╠═25a5c232-a0f6-4a1b-8f91-3cee5d97b3db
# ╠═75e8312d-1614-4626-99d1-3e07fdec8bab
# ╠═3d1f7cfa-ae9b-4ed1-9261-595fee65a974
# ╠═fa50cac7-64f9-4439-9645-c6c0b4fa0b99
# ╠═28ec1019-f51e-4dd9-a5ca-cfd0a095fcea
# ╟─2045e894-396b-44d3-8dd8-ff0b88c598c2
# ╠═86683b41-7b2a-471f-b817-bc0925c3770f
# ╠═493328f4-fcfd-49d4-84f7-6540ea6d2319
# ╠═72101250-16ae-4c20-a7aa-e1479a41860f
# ╟─58b6234b-a10e-46de-8031-ab65028d3c89
# ╠═8bd8627b-9884-4efd-acf9-e1e57a52c833
# ╠═06eabbc3-94af-4b76-9369-9e5df468cbbc
# ╠═494521db-50e9-48ee-913e-1ac3c70d3172
# ╟─380f9d36-fb68-490a-9021-16c265056a3e
# ╠═c63f9137-574d-4c70-91ce-99fdc304bf02
# ╠═1b3230db-bdda-4c71-9b33-31c779d50a7b
# ╠═7242a8c3-362c-4df9-b518-1a2a40ca2f39
