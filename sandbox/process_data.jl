### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e4d22a48-8b16-4e7a-a776-fd12b6023123
begin
	using CSV, DataFrames, JLD2, ZipFile
	using Graphs, LinearAlgebra, MolecularGraph, MolecularGraphKernels
	using Cairo, MolecularGraph, Rsvg
	using CairoMakie, PlutoUI
	using PyCall, Statistics
	import AlgebraOfGraphics as AOG
	TableOfContents()
end

# ╔═╡ d5a773d7-3718-4cd1-be2a-68ad929bf051
begin
	AOG.set_aog_theme!(fonts=[AOG.firasans("Light"), AOG.firasans("Light")])
	update_theme!(fontsize=20, linewidth=3, resolution=(500, 400))
	viz_theme = true
end;

# ╔═╡ 27ae40bf-6631-4737-9a3c-42a3de65964c
md"# Olfactory Perception Data"

# ╔═╡ e9799791-107d-4693-ae34-767a56d302c0
md"""
Combine and pre-process the olfactory perception data of _Goodscents_ and _Leffingwell_, 
as obtained from [_The Pyrfume Project_](https://pyrfume.org/); mimic the procedure 
employed in [\"Machine Learning for Scent: Learning Generalizable Perceptual 
Representations of Small Molecules\"](https://arxiv.org/abs/1910.10685). 
"""

# ╔═╡ 27791708-984b-4ac1-bc70-fdfd97fe708e
md"""
Use the following commit on the `pyrfume-data` repository to download the specific version of the data (most recent as of Jan. 2023)
"""

# ╔═╡ e1c1ead9-bebc-462e-9209-acb2aed030c5
pyrfume_commit = "41a38b0657da131123a4299f4d2295b0d7e81ecb"

# ╔═╡ 2711feb5-a6e5-4172-959d-60be301bdb80
md"""
## Leffingwell Data
"""

# ╔═╡ 8926fafe-c057-43e0-bd8b-849e9af45ba7
leffingwell_data = CSV.read(
	download("https://raw.githubusercontent.com/pyrfume/pyrfume-data/$pyrfume_commit/leffingwell/behavior.csv"), 
	DataFrame
)

# ╔═╡ d5f1e207-0d57-476a-af6f-aa960fc55bd0
md"""
get the odor label column names
"""

# ╔═╡ 39974a37-f5f1-48dc-b3e9-82c80583a570
leffingwell_odors = names(leffingwell_data)[3:end]

# ╔═╡ 1cea56f6-4cf6-4005-a696-72ee77fadefb
length(leffingwell_odors) # number of unique 

# ╔═╡ 3d7a3459-afcd-45ba-a1e6-547d09226019
setdiff(reduce(vcat, split.(leffingwell_odors)), leffingwell_odors)

# ╔═╡ d0bdc461-c415-4271-87be-294688883c8a
setdiff(leffingwell_odors, reduce(vcat, split.(leffingwell_odors)))

# ╔═╡ 439855bd-b35b-4f89-9a26-9ed670d48b16
md"""
### 🚩
!!! note "Leffingwell Multi-Word Labels"
	There is one: "black currant"
"""

# ╔═╡ 4cefde35-c896-430f-93bc-0b759c1f955d
md"""
convert the bit encoding from each row into a list of odors
"""

# ╔═╡ 4e8566ac-bdd2-4842-8f72-28f2ea4c625f
leffingwell_labels = [
	[odor for odor in leffingwell_odors if row[odor] == 1] 
	for row in eachrow(leffingwell_data)
]

# ╔═╡ 5c734e51-2ecd-4a38-b574-381fa88d52c9
md"""
make dataframe w/ SMILES and odors
"""

# ╔═╡ d539b75a-c1e6-4c71-933d-443fc2133330
leffingwell_df = DataFrame([
	"molecule" => leffingwell_data[:, "IsomericSMILES"]
	"odor (Leffingwell)" => leffingwell_labels
])

# ╔═╡ 1df38111-f1a0-43c9-a2d5-6815100536c7
md"""
check to make sure molecules are not repeated
"""

# ╔═╡ f10eb262-c858-4269-80eb-d7344fd3e0c4
@assert length(unique(leffingwell_df.molecule)) == nrow(leffingwell_df)

# ╔═╡ fbe99ab0-c9af-4a8d-bbb7-708cbf674016
md"check to make sure labels are unique"

# ╔═╡ 51c34593-b29e-4e9e-96b4-bf2105ecae13
@assert all(length.(leffingwell_df[:, "odor (Leffingwell)"]) .== length.(unique.(leffingwell_df[:, "odor (Leffingwell)"])))

# ╔═╡ fb9db5bd-9b94-4784-b91f-baae997d18b0
md"""
## Goodscents Data
"""

# ╔═╡ 4e0695f8-eff9-4709-9d74-841e97c2f5ad
md"""
three files in this data set
"""

# ╔═╡ 5c104d70-8bad-4fd7-a5db-39198e23efe1
goodscents_stimuli = CSV.read(
	download("https://raw.githubusercontent.com/pyrfume/pyrfume-data/$pyrfume_commit/goodscents/stimuli.csv"), 
	DataFrame
)

# ╔═╡ a76159c9-c4d0-4e3b-b25d-58fc2904edfa
goodscents_behavior = CSV.read(
	download("https://raw.githubusercontent.com/pyrfume/pyrfume-data/$pyrfume_commit/goodscents/behavior.csv"), 
	DataFrame
)

# ╔═╡ 411255f2-2b8b-4a9b-9f4b-d3320541c702
all_gs_strings = reduce(
	vcat, 
	[split(x, ';') for x in goodscents_behavior.Descriptors if !ismissing(x)]
) |> unique

# ╔═╡ 6e9a4254-6e55-4527-9db5-b9049d320428
gs_single_words = reduce(vcat, split.(all_gs_strings)) |> unique

# ╔═╡ b57ca630-1bf8-4275-a583-09e257a670a2
setdiff(all_gs_strings, gs_single_words)

# ╔═╡ 0fb36c69-0c34-4c03-93b5-3ad1b487559b
setdiff(gs_single_words, all_gs_strings)

# ╔═╡ 491b2929-1c6b-41c4-b9c4-b155e4737d65
goodscents_molecules = CSV.read(
	download("https://raw.githubusercontent.com/pyrfume/pyrfume-data/$pyrfume_commit/goodscents/molecules.csv"), 
	DataFrame
)

# ╔═╡ 8dee36a8-82ab-41b5-9ec5-7256efdfa5f5
md"""
join the 3 dataframes on the "Stimulus" and "CID" keys
"""

# ╔═╡ b2ececfd-69fc-41e6-a169-dc4ff1245373
goodscents_df1 = innerjoin(
	innerjoin(goodscents_behavior, goodscents_stimuli, on="Stimulus"), 
	goodscents_molecules, 
	on="CID"
)

# ╔═╡ 8cff11cf-60fa-4ce5-ad08-82075b00c308
md"""
drop extraneous columns, and rename columns
"""

# ╔═╡ cd0f7408-5883-498f-b040-f68fbbdcd4bc
goodscents_df2 = begin
	local df = select(goodscents_df1, ["IsomericSMILES", "Descriptors"])
	rename!(df, "IsomericSMILES" => "molecule")
	rename!(df, "Descriptors" => "odor (Goodscents)")
	df
end

# ╔═╡ 31d7054e-ecab-4553-8588-30cbd4255b15
md"""
drop rows with no odor data
"""

# ╔═╡ 3d67fd30-4c95-4f46-abab-daee2b155a08
goodscents_df3 = filter(row -> !ismissing(row["odor (Goodscents)"]), goodscents_df2)

# ╔═╡ 5aa3c3ea-e862-440b-92ed-7a0415f6ec5e
md"""
split odor strings into arrays
"""

# ╔═╡ 93703e72-8718-44f1-8998-f38799efccce
goodscents_df4 = transform(
	goodscents_df3, 
	"odor (Goodscents)" => (
		col -> split.(col, ';')
	); 
	renamecols=false
)

# ╔═╡ f2569c11-afe0-429a-a479-06a674936128
md"""
some molecules are repeated. this merges them and takes the union of their scent labels.
"""

# ╔═╡ cf866228-5674-481e-b9df-2bad986bc46c
goodscents_df = combine(
	groupby(goodscents_df4, "molecule"),
	"odor (Goodscents)" => (col -> [unique(vcat(col...))]),
	renamecols=false
)

# ╔═╡ 270b67f4-94d4-4008-957c-afa8084f5082
md"""
check to make sure molecules are not repeated
"""

# ╔═╡ c6b10ccd-2fa9-4fbb-aa59-8ad32fadf2b6
@assert length(unique(goodscents_df.molecule)) == nrow(goodscents_df)

# ╔═╡ 24d34dc0-9b78-41dc-add0-8772a970b31c
md"check to make sure labels are unique"

# ╔═╡ 6ee51a94-8449-4fe7-896b-13507d9b3f8b
@assert all(length.(goodscents_df[:, "odor (Goodscents)"]) .== length.(unique.(goodscents_df[:, "odor (Goodscents)"])))

# ╔═╡ 308e3f12-00c2-4117-b4e7-8b469344189c
md"""
## Combine
"""

# ╔═╡ e94ab1f3-4394-40b6-8223-10a3b7707606
md"""
join the two datasets on molecule
"""

# ╔═╡ 380a90ae-a0ae-445e-83a1-40d7ed85fafb
combined_df1 = outerjoin(goodscents_df, leffingwell_df; on=:molecule)

# ╔═╡ 309f21a9-47c6-469f-9ca9-62a7b5b9dfba
md"""
replace missing values with empty array
"""

# ╔═╡ 1a7d0861-ec0d-4e8c-890e-48fe69c612c5
combined_df2 = begin
	local df = transform(
		combined_df1, 
		"odor (Goodscents)" => (col -> [ismissing(row) ? [] : row for row in col]); 
		renamecols=false
	)
	transform!(
		df, 
		"odor (Leffingwell)" => (col -> [ismissing(row) ? [] : row for row in col]); 
		renamecols=false
	)
	df
end

# ╔═╡ 9db8dbda-59cd-4e2a-ad8a-bf2978bedaba
md"""
take union of labels for each molecule and drop old columns
"""

# ╔═╡ a0f25728-d129-405d-b715-db85f269a7d6
combined_df3 = begin
	local df = transform(
		combined_df2,
		["odor (Goodscents)", "odor (Leffingwell)"] =>
		((col1, col2) -> col1 .∪ col2) =>
		"odor"
	)
	select!(df, ["molecule", "odor"])
	df
end

# ╔═╡ bc402deb-5889-4a0d-b30c-d0e9295cb66e
md"""
eliminate one row (divinyl sulfide) because the labels are all unique to just this one molecule
"""

# ╔═╡ 9b4e8cf6-8fe3-4022-870a-09f0d7933be5
# divinyl sulfide only has totally unique labels
combined_df4 = filter(r -> r.molecule ≠ "C=CSC=C", combined_df3)

# ╔═╡ c976895a-f99e-478e-a44b-298fc8c841f2
md"""
## Fix ABA Labels
"""

# ╔═╡ 65986bc4-9f99-4e6f-af3e-a8989aa0f75a
md"""
Rectify "A B A" to "A", "B A"
"""

# ╔═╡ b1ac9d4f-dd01-4494-a9d8-a5d212ab366a
function detect_ABA(str)
	substrings = split(str)
	length(substrings) == 3 || return false
	A, B, A′ = substrings
	return A == A′
end

# ╔═╡ 9cea4488-3e06-46b9-b78b-2c971ddd2a24
has_ABA = [
	any(detect_ABA(label) for label in row.odor) for row in eachrow(combined_df4)
]

# ╔═╡ d51c2ae4-db6d-4d91-9f22-3a82deb74a6d
function rectify(str)
	A, B, _ = split(str)
	return [A, "$B $A"]
end

# ╔═╡ 83d308e1-f098-4d32-8e94-10d1db5a2eaa
function fix_ABA(vec::Vector)
	# check for the ABA condition
	w_ABA = detect_ABA.(vec)
	if !any(w_ABA)
		# not found, make no changes
		return vec
	end
	wo_ABA = .! w_ABA
	str_wo_ABA = vec[wo_ABA] # non-ABA strings to leave as they are
	str_w_ABA = vec[w_ABA] # ABA strings to rectify
	rectified = reduce(vcat, rectify.(str_w_ABA)) # "A B A" -> ["A", "B A"]
	return vcat(str_wo_ABA, rectified) |> unique # enforce uniqueness
end

# ╔═╡ 6e015806-c499-4439-bcbd-db57bca1d5a6
fix_ABA(combined_df4.odor[1])

# ╔═╡ 8c95e70f-fbf2-4da4-a23a-30984ce03c9b
length(combined_df4.odor[1])

# ╔═╡ 796179c4-851f-485e-ba11-02c57bcd6b3c
combined_df = begin
	local df = transform(combined_df4, r -> fix_ABA.(r.odor))
	odor = df.x1
	DataFrame(:molecule => df.molecule, :odor => odor)
end

# ╔═╡ c384c30e-0b7e-4759-92ff-09b00f798613
pyrfume_odor_labels = unique(reduce(vcat, combined_df.odor))

# ╔═╡ 1840b5b4-dbcb-4ba2-a3ad-d97116619689
reencoding = Dict([
	"alliaceous" 					=> ["garlic"]
	"caramellic" 					=> ["caramel"]
	"coumarinic" 					=> ["coumarin"]
	"lactonic" 						=> ["lactone"]
	"camphoreous" 					=> ["camphor"]
	"mentholic" 					=> ["menthol"]
	"black currant" 				=> ["blackcurrant"]
	"thujonic" 						=> ["thujone"]
	"dihydrojasmonate" 				=> ["jasmine"]
	"cornmint" 						=> ["mint"]
	"guaiacwood" 					=> ["guaiacol"]
	"muttony" 						=> ["mutton"]
	"lilial" 						=> ["lily"]
	"tagette" 						=> ["taget"]
	"citralva" 						=> ["citrus"]
	"privetblossom" 				=> ["privet"]
	"nitromusk" 					=> ["musk"]
	"hesperidic" 					=> ["citrus"]
	"chavicol" 						=> ["phenolic"]
	"woody-lactone" 				=> ["woody", "lactone"]
	"heptine" 						=> ["cucumber"]
	"verdox" 						=> ["apple"]
	"damascone" 					=> ["rose"]
	"violet leaf" 					=> ["violet"]
	"apple skin" 					=> ["apple"]
	"bois de rose" 					=> ["rose"]
	"juicy fruit" 					=> ["fruit"]
	"melon rind" 					=> ["melon"]
	"root beer" 					=> ["rootbeer"]
	"grapefruit peel" 				=> ["grapefruit"]
	"egg nog" 						=> ["eggnog"]
	"bubble gum" 					=> ["bubblegum"]
	"tutti frutti" 					=> ["fruit"]
	"lemon peel" 					=> ["lemon"]
	"watermelon rind" 				=> ["watermelon"]
	"banana peel" 					=> ["banana"]
	"nut skin" 						=> ["nutty"]
	"grape skin" 					=> ["grape"]
	"corn chip" 					=> ["corn"]
	"cheesy parmesan cheese" 		=> ["parmesan", "cheese"]
	"cheesy bleu cheese" 			=> ["bleu", "cheese"]
	"orange peel" 					=> ["orangepeel", "orange"]
	"citrus rind" 					=> ["citrus"]
	"orange rind" 					=> ["orangepeel", "orange"]
	"fresh outdoors" 				=> ["outdoors", "fresh"]
	"cheesy roquefort cheese" 		=> ["roquefort", "cheese"]
	"cotton candy" 					=> ["cottoncandy", "candy"]
	"bread baked" 					=> ["toasty", "bread", "baked"]
	"citrus peel" 					=> ["citrus"]
	"graham cracker" 				=> ["graham", "cracker"]
	"currant bud black currant bud" => ["blackcurrant"]
	"passion fruit" 				=> ["passionfruit"]
	"beef juice" 					=> ["beef"]
	"peanut butter" 				=> ["peanutbutter", "peanut"]
	"cheesy feta cheese" 			=> ["feta", "cheese"]
	"lily of the valley" 			=> ["lily"]
	"hay new mown hay" 				=> ["hay"]
	"tomato leaf" 					=> ["tomato"]
	"butter rancid" 				=> ["rancid", "butter"]
	"bread crust" 					=> ["bread", "crust"]
	"woody burnt wood" 				=> ["woody", "burnt", "wood"]
	"nut flesh" 					=> ["nutty"]
	"chicken fat" 					=> ["chicken", "fat"]
	"potato chip" 					=> ["potato"]
	"carrot seed" 					=> ["carrot"]
	"cucumber skin" 				=> ["cucumber"]
	"cheesy limburger cheese" 		=> ["limburger", "cheese"]
	"woody old wood" 				=> ["woody", "old", "wood"]
	"sweet pea" 					=> ["pea", "sweet"]
	"chicken coup" 					=> ["droppings", "chicken", "coop"]
	"linden flower" 				=> ["linden", "flower"]
	"fir needle" 					=> ["fir"]
	"plum skin" 					=> ["plum"]
	"pear skin" 					=> ["pear"]
	"valerian root" 				=> ["valerian"]
	"egg yolk" 						=> ["eggyolk", "egg", "yolk"]
	"almost odorless" 				=> ["odorless"]
	"tolu balsam" 					=> ["cinnamon"]
	"new mown hay" 					=> ["hay"]
	"methyl heptine carbonate" 		=> ["cucumber"]
	"concord grape" 				=> ["grape"]
])

# ╔═╡ bb2ed649-e2bb-4f27-a16f-a071fcb498f4
md"""
re-label odors according to scheme
"""

# ╔═╡ 1462bed5-ce9f-4969-bd99-f91fde1ba9e9
relabeled_df1 = transform(
	combined_df,
	"odor" => (
		col -> 
			map(
				row -> reduce(
					vcat, 
					[
						odor in keys(reencoding) ? reencoding[odor] : odor
						for odor in row
					]
				), 
				col
			)
	);
	renamecols=false
)

# ╔═╡ c7aaa8a8-8b7a-4d6e-b1d8-73a6ef7ca559
md"""
some molecules have single odor labels; make sure they are wrapped in `Vector`s
"""

# ╔═╡ c76d435d-5e96-46a1-97ab-91e6d88f23cf
relabeled_df = transform(
	relabeled_df1,
	"odor" => (
		col -> map(row -> eltype(row) <: AbstractString ? row : [row], col)
	);
	renamecols=false
)

# ╔═╡ 05a38bf7-ea97-4d68-9381-db7c9488d29d
md"make odors unique"

# ╔═╡ 0ac3ff6b-8123-4b40-939d-2e6a292997d4
unique_label_df = begin
	unique_label_df = transform(
		relabeled_df, "odor" => col -> unique.(col), 
		renamecols=false
	)
	unique_label_df[:, "# odors"] = length.(unique_label_df[:, "odor"])
	unique_label_df
end

# ╔═╡ 8714f0f2-8480-40cb-93c0-7c187739ec6e
md"""
## Exclude Odorless Inorganic Salts
"""

# ╔═╡ 812a9f11-b2e0-4fa6-ac23-f77b7aee055a
md"""
some substances have zero graph motifs in common with any others; other substances share graph motifs only within a small group, such that the molecule kernel graph is not fully connected. these all turn out to be odorless inorganic salts, so we will exclude them.
"""

# ╔═╡ fd343574-2637-480e-9396-e3d1616ae8a9
salts_to_exclude = [
	# no bonds in common with anything:
	"O=[Ca]"
	"O=[Cr]O[Cr]=O"
	"O=[Mg]"
	"O=[Zn]"
	"O=[V](=O)O[V](=O)=O"
	"O=[Cu]"
	"O=[Ti]=O"
	"O=[Bi].Cl"
	# only shares features w/ each other (iodates):
	"[O-]I(=O)=O.[K+]"
	"[O-]I(=O)=O.[O-]I(=O)=O.[Ca+2]"
	# only shares features w/ each other (silicates):
	"O=[Si]=O"
	"[O-][Si]([O-])([O-])[O-].[Mg+2].[Al+3]"
	"[O-][Si](=O)[O-].[Mg+2]"
	"[O-][Si](=O)[O-].[O-][Si](=O)[O-].[Na+].[Al+3]"
	"[O-][Si]([O-])([O-])[O-].[Ca+2].[Ca+2]"
	"O[Si](=O)[O-].O[Si](=O)[O-].[O-][Si](=O)[O-].[O-][Si](=O)[O-].[Mg+2].[Mg+2].[Mg+2]"
	"C[Si]1(O[Si](O[Si](O[Si](O[Si](O[Si](O1)(C)C)(C)C)(C)C)(C)C)(C)C)C"
]

# ╔═╡ cc941a57-7ce9-4007-8a82-fa16744a4b45
salt_idx = [
	findfirst(isequal(salt), unique_label_df.molecule) for salt in salts_to_exclude
]

# ╔═╡ c024aa88-2850-436e-a830-ced1bd36807f
unique_label_df[salt_idx, :]

# ╔═╡ 8a787b8b-31e0-4928-bdba-a1afb61a1411
md"""
some molecules' graphs have no edges, due to SMILES conventions used and the choice to not include explicit hydrogens
"""

# ╔═╡ 6cba5e1a-31d5-4204-a5f1-78bd8e90dd18
mol_has_bonds(df_row) = nv(MetaGraph(smilestomol(df_row.molecule))) ≠ 0

# ╔═╡ 64f71860-501c-4db0-b498-bfb2477dc912
no_isolated_atoms(df_row) = !any(
	isequal(0), 
	degree(MetaGraph(smilestomol(df_row.molecule)))
)

# ╔═╡ 61ff9250-07d9-459f-8f98-3572199529c4
md"""
filter out problematic molecules (divinyl sulfide, odorless salts, isolated atoms)
"""

# ╔═╡ e45054ad-d11c-4eef-8623-504096934f53
filtered_df = begin
	local df = copy(unique_label_df)
	df = filter(row -> row.molecule ∉ salts_to_exclude, df)
	df = filter(mol_has_bonds, df)
	df = filter(no_isolated_atoms, df)
	df
end

# ╔═╡ fbfe1bb0-80d1-4d4b-90f9-af60ab512ae4
function counts_per_label(df)
	labels = reduce(vcat, df.odor)
	counts = Dict()
	for label in unique(labels)
		counts[label] = count(isequal(label), labels)
	end
	return counts
end

# ╔═╡ cb5a017e-6038-48fc-9ec0-49f0eee745d4
md"""
remove odor labels that occur less than 30 times
"""

# ╔═╡ 9d4ff5db-7dc4-48e3-af8f-dc8e42914c78
final_df1 = begin
	local df = copy(filtered_df)
	local cpl = counts_per_label(df)
	local low_count_labels = [label for label in keys(cpl) if cpl[label] < 30]
	local new_odor_col = copy(df.odor)
	for (i, label_vec) in enumerate(df.odor)
		new_odor_col[i] = [label for label in label_vec if label ∉ low_count_labels]
	end
	df.odor .= new_odor_col
	df
end

# ╔═╡ f5a9cde7-ef6c-4b81-9656-78727d9dd0ba
md"""
remove molecules that had all their labels removed
"""

# ╔═╡ 64cfae63-0110-44dc-9335-c90d7d3a4370
final_df = begin
	local df = copy(final_df1)
	local zero_label_rows = [length(row.odor) == 0 for row in eachrow(df)]
	df[.! zero_label_rows, :]
end

# ╔═╡ 389b64ed-71b7-482b-a7c4-9ae392642a83
@assert all(length.(final_df[:, "odor"]) .== length.(unique.(final_df[:, "odor"])))

# ╔═╡ 9654fe84-8bbb-4937-8d7a-b9f0b7c1949b
md"""
collect new list of labels
"""

# ╔═╡ 478b57a5-b74d-4142-bd55-6f73b9c076f1
odor_labels = unique(reduce(vcat, final_df.odor))

# ╔═╡ 807c257c-24fe-4c4b-9965-03a8a58ff271
md"""
## Export Data
"""

# ╔═╡ 3855a083-851a-41ec-9e42-1864c5e90b76
md"""
write the odor reencoding scheme, list of all odor labels, and combined dataframe to disk
"""

# ╔═╡ f8294d1c-7a68-4196-ac7d-f78b00567d77
begin
	@save "data/reencoding.jld2" reencoding
	CSV.write("data/pyrfume.csv", final_df)
	open("data/odor_labels.txt", "w") do f
        write(f, reduce(*, odor_labels .* "\n"))
    end
end;

# ╔═╡ 964f1749-67e4-4f17-b6f5-2efc5017d4d3
md"label matrix"

# ╔═╡ 2f5e7378-b8cc-44ba-8eb8-c1ab85e92f79
begin
	label_matrix = zeros(Int, nrow(final_df), length(odor_labels))
	for (i, odors) in enumerate(final_df[:, "odor"])
	    for (j, label) in enumerate(odor_labels)
	        if label in odors
	            label_matrix[i, j] = 1
	        end
	    end
	end
	label_matrix
end

# ╔═╡ 7cbe8e41-2571-4ea7-a359-6d2dbb87030a
@save "data/label_matrix.jld2" label_matrix

# ╔═╡ b59bd9a3-cc90-4bf9-ab42-6ea424df8f2d
@assert sum(label_matrix) == sum([length(o) for o in final_df[:, "odor"]])

# ╔═╡ 4b6dfaac-6d81-4343-8b5c-c40dd6e39129
md"""
## Statistics, Visualization
"""

# ╔═╡ d20ffd27-335f-4e16-9f50-b48552bd6b22
md"""
count the instances of each label
"""

# ╔═╡ 1eafe46f-1bfb-4cb1-b4bd-ef6b308f11bc
label_counts = sum(label_matrix, dims=1)[:]

# ╔═╡ 95b5c3a3-1948-4ac9-a665-8103cf0afd66
@assert ! any(label_counts .== 0)

# ╔═╡ f8a2177a-efb5-43a3-9649-ef030429506d
md"""
sort, descending, by count
"""

# ╔═╡ 054fd209-287f-4e77-bbbc-7dd32e5a08aa
sp = sortperm(label_counts; rev=true)

# ╔═╡ 8ec91912-31b8-40dc-ac6c-f9026bb5ccb6
md"""
enumerate the label frequencies
"""

# ╔═╡ 924dbe68-af24-4e7a-b285-84b4f22fbbb0
odor_prevalances = DataFrame(
	"odor"        => odor_labels[sp], 
	"# molecules" => label_counts[sp]
)

# ╔═╡ 26895aa5-d6e0-48e2-8811-bb7378d3e74d
begin
	local op = odor_prevalances[1:15, :]
	local fig = Figure()#resolution=(500, 11000))
	local ax  = Axis(fig[1, 1], 
		xlabel="# molecules", 
		ylabel="odor", 
		title="# molecules per odor label (top 15)",
		yticks=(1:nrow(op), op[:, "odor"])
	)
	barplot!(1:nrow(op), op[:, "# molecules"], direction=:x)
	ylims!(-0.5, nrow(op)+0.5)
	fig
end

# ╔═╡ b5647646-b043-42e3-b104-5c1537c2970d
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1]; title="# molecules per odor label", xlabel="# molecules", ylabel="Frequency")
	hist!(ax, odor_prevalances[:, "# molecules"]; normalization=:probability, bins=50)
	fig
end

# ╔═╡ 81e3d848-3808-4af2-874a-8f38f1a79545
odor_label_counts = combine(groupby(final_df, "# odors"), nrow => "# molecules")

# ╔═╡ ba788b21-ba0d-4a4a-9463-8778124240c7
begin
	local fig = Figure(resolution=(600, 400))
	local ax  = Axis(fig[1, 1],
		xlabel="# odor labels",
		ylabel="# molecules",
		xticks=1:25,
		title="# odor labels per molecule"
	)
	barplot!(odor_label_counts[:, "# odors"], 
		     odor_label_counts[:, "# molecules"])
	fig
end

# ╔═╡ 813d9850-f19a-49f0-bced-c71c50ba5b6c
md"""
\# of molecules
"""

# ╔═╡ 39c91036-b8a5-4082-97ee-1093e3fa4b28
nrow(final_df)

# ╔═╡ cc536782-4512-4786-b6af-ee6ce269931e
md"\# molecules in leffingwell/goodscents"

# ╔═╡ 2c3f3a16-98bf-4ed4-949a-9c46de02b3d2
nrow(leffingwell_df)

# ╔═╡ 17649e4a-657d-468d-a393-fba0f4d881d4
nrow(goodscents_df)

# ╔═╡ 72cbc15d-59ee-49c7-a9d4-d5e3cdea9cd2
md"\# molecules in intersection"

# ╔═╡ 4c1d1736-093d-4716-8317-6bd2072e6a82
nrow(leffingwell_df) + nrow(goodscents_df) - nrow(final_df)

# ╔═╡ f05cfa94-0807-4305-8159-4836342bb161
md"# Molecules"

# ╔═╡ 9185f35e-c3b8-4639-8b87-d6f60bcf9fe2
md"""
## Build Graphs

convert from SMILES to graph
"""

# ╔═╡ 30318671-b64f-4675-9954-3ce6953cf990
graphs = MetaGraph.(smilestomol.(final_df[:, "molecule"]))

# ╔═╡ 074d3e61-3c3b-4b20-9d85-7f270b354c4d
md"""
## Visualize
"""

# ╔═╡ e4a0812b-7747-4c17-a57a-6d7247106fdf
id_random_molecule = rand(eachindex(graphs))

# ╔═╡ b6870321-407c-46d8-8c83-d8a3ee187cfe
combined_df3[id_random_molecule, "molecule"]

# ╔═╡ ce3a40ba-c942-4d92-8903-7f1906aab08d
viz_graph(graphs[id_random_molecule]; layout_style=:molecular)

# ╔═╡ 16ca287c-788e-4d87-8f16-1db41dda872f
HTML(drawsvg(smilestomol(final_df[id_random_molecule, "molecule"]), 500, 500))

# ╔═╡ 60a734a8-53ba-4908-b6c6-bccc9c93a938
function mol_to_img(smiles::AbstractString, id::Int=0; x::Int=250, y::Int=250)
	tempfile = tempname()

	mol = smilestomol(smiles)
	s = drawsvg(mol, x, y)
	r = Rsvg.handle_new_from_data(s)
	d = Rsvg.handle_get_dimensions(r)
	cs = Cairo.CairoImageSurface(d.width, d.height, Cairo.FORMAT_ARGB32)
	c = Cairo.CairoContext(cs)
	Rsvg.handle_render_cairo(c, r)
	Cairo.write_to_png(cs, tempfile)

	if !isdir("data/img")
		mkdir("data/img")
	end
	cp(tempfile, "data/img/$id.png")
	img = CairoMakie.image(rotr90(load(tempfile)))
	
	return img
end;

# ╔═╡ 33026be4-fa61-4202-a3cf-d238273192a0
begin
	rm("data/img"; recursive=true, force=true)
	mol_imgs = mol_to_img.(final_df.molecule, 1:nrow(final_df))
end;

# ╔═╡ 83d9bc3b-cdeb-4b84-a8d3-c1cca85affea
mol_imgs[id_random_molecule]

# ╔═╡ 13301f55-094f-4f59-b0ed-faac030d090b
md"""
## Export
"""

# ╔═╡ e46ce0f0-76bc-4cac-83fd-3573ed62fd98
md"""
Re-compute graphs from final dataframe SMILES
"""

# ╔═╡ d0832aad-4242-4b66-aae5-a8b5f09a46a0
@assert nrow(final_df) == size(label_matrix, 1)

# ╔═╡ 264f1fe3-62e1-4248-8129-d4d3c14af8ff
@save "data/molecule_graphs.jld2" graphs

# ╔═╡ 1a5c3d6c-a941-4511-80c5-b1c67dcf8bb4
# images saved in data/img

# ╔═╡ be970511-03a3-4844-84b6-141e375ca99f
md"# Odor Gramian


"

# ╔═╡ 356fc6a8-3363-48da-a802-494a090b120a
md"""
## Get BERT Embeddings
"""

# ╔═╡ cb00cedc-577b-44d1-aa78-3ba17083bdc0
begin
	transformers = pyimport("transformers")
	pretrained_model = "bert-base-uncased"
	tokenizer = transformers.BertTokenizerFast.from_pretrained(pretrained_model)
	model = transformers.BertModel.from_pretrained(pretrained_model)
end;

# ╔═╡ dbe3b4e3-1333-4f78-b22f-9a434607b9fc
prompts = [
	"\$x",
	"does this smell like \$x?",
	"this smells of \$x",
	"i detect a strong odor of \$x",
	"does this cause the distinct smell of \$x?",
	"this scent reminds me of \$x",
	"do you smell a hint of \$x in the air?",
	"i smell a mixture of scents, and \$x is one of them",
	"smell this sample. does it match the description of \"\$x?\""
];

# ╔═╡ a4115745-8ab0-4231-9372-c130b3f8a4c1
generate_prompt(prompt, s) = replace(prompt, "\$x" => "$s")

# ╔═╡ 52282b6e-19de-4a2d-9258-be317a0b4d71
function encode_with_bert(s::AbstractString)
	py"""	
	inputs = $tokenizer($s, return_tensors="pt")
	"""
	inputs = py"inputs"
	k = inputs[:word_ids]()[2:end-1] .+ 1 |> x -> findfirst(x .== maximum(x)) + 1
	py"""
	outputs = $model(**$inputs, output_hidden_states=True)
	"""
	outputs = py"outputs"
	h = [hₜ.detach().numpy()[1, :, :] for hₜ in outputs["hidden_states"]]
	return mean(reduce(hcat, [x[k, :] for x in h]); dims=2)
end

# ╔═╡ eb3afc91-4207-44a1-a97f-8306ad5d6d75
encodings = Dict(
	p => Dict(
		odor_labels .=> [encode_with_bert(generate_prompt(p, l)) for l in odor_labels]
	) 
	for p in prompts
)

# ╔═╡ d92fc1dc-ae87-49d9-94e9-872f8c02f581
md"## Construct Matrix"

# ╔═╡ a8dea5db-e306-47df-a2c7-a1da39e25b9b
function make_odor_gramian(prompt)
	odor_gramian = zeros(length(odor_labels), length(odor_labels))
	for (j, word_j) in enumerate(odor_labels)
	    bert_j = encodings[prompt][word_j]
	    for (i, word_i) in enumerate(odor_labels)
	        if i > j
	            break
	        end
	        bert_i = encodings[prompt][word_i]
	        odor_gramian[i, j] = odor_gramian[j, i] =
	            dot(bert_i, bert_j) / norm(bert_i) / norm(bert_j)
	    end
	end
	odor_gramian .+= abs(minimum(odor_gramian))
	odor_gramian ./= maximum(odor_gramian)
end

# ╔═╡ 24e4487b-8117-43b1-bd09-feb2f38fe1bb
odor_gramian = Dict(
	prompt => make_odor_gramian(prompt) for prompt in keys(encodings)
)

# ╔═╡ 1eb5ca97-bb40-4553-8c21-e3315230ecfc
md"""
## Export
"""

# ╔═╡ a3098e7b-c85d-40ac-a6f1-3f1c2f9f85da
@save "data/odor_gramian.jld2" odor_gramian

# ╔═╡ 6467582e-837d-4c73-a168-054fe6e96fc8
jldsave("data/bert_encodings.jld2"; encodings)

# ╔═╡ 9e8a1722-1bee-42f7-a426-2a188e53582a
md"""
## Visualization
"""

# ╔═╡ e547a268-6188-471b-ab9e-5e5fc8433e1d
@bind viz_prompt_select Select(collect(keys(odor_gramian)))

# ╔═╡ a0be8244-930b-40f4-a1d5-16cdaa8161d0
begin
	local fig = Figure()
	local ax = Axis(fig[1, 1]; aspect=AxisAspect(1), title=viz_prompt_select)
	hm = heatmap!(ax, reverse(odor_gramian[viz_prompt_select]; dims=2); colormap=:viridis)
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 3284de1e-a1cb-4ed9-90a4-c3bfa535b3a6
hist(odor_gramian[viz_prompt_select][:]; normalization=:probability)

# ╔═╡ f2fa0edc-4633-4781-b032-7e7257bd63eb
md"""
# Summary of Data
* `odor_labels` is ordered list of odor labels.
* `final_df[:, \"molecule\"]` is ordered list of molecules.
* `graphs` (as exported) contains graphs in same order as `df[:, \"molecule\"]`
* `label_matrix` connects molecules with odors (0 or 1)
* `odor_gramian` gives similarity of odors
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Cairo = "159f3aea-2a34-519c-b102-8c37f9878175"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MolecularGraph = "6c89ec66-9cd8-5372-9f91-fabc50dd27fd"
MolecularGraphKernels = "bf3818bd-b6bb-4954-8baa-32c32282e633"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
Rsvg = "c4c386cf-5103-5370-be45-f3a111cca3b8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
ZipFile = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"

[compat]
AlgebraOfGraphics = "~0.6.14"
CSV = "~0.10.9"
Cairo = "~1.0.5"
CairoMakie = "~0.10.2"
DataFrames = "~1.4.4"
Graphs = "~1.8.0"
JLD2 = "~0.4.30"
MolecularGraph = "~0.12.0"
MolecularGraphKernels = "~0.8.5"
PlutoUI = "~0.7.49"
PyCall = "~1.96.1"
Rsvg = "~1.0.0"
ZipFile = "~0.10.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "ae7afb2a4b807262cce8cd958ba0f2f5206527e4"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0310e08cb19f5da31d08341c6120c047598f5b9c"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.5.0"

[[deps.AlgebraOfGraphics]]
deps = ["Colors", "Dates", "Dictionaries", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "KernelDensity", "Loess", "Makie", "PlotUtils", "PooledArrays", "RelocatableFolders", "SnoopPrecompile", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "43c2ef89ca0cdaf77373401a989abae4410c7b8a"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.6.14"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.Aqua]]
deps = ["Compat", "Pkg", "Test"]
git-tree-sha1 = "cee4fc289106df4d2d7f25f3918211b271e38bb0"
uuid = "4c88cf16-eb10-579e-8560-4a9242c79595"
version = "0.5.6"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "SnoopPrecompile", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "c700cce799b51c9045473de751e9319bdd1c6e94"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.9"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "SnoopPrecompile"]
git-tree-sha1 = "abb7df708fe1335367518659989627100a61f3f0"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.10.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "d853e57661ba3a57abcdaa201f4c9917a93487a2"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.4"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "8c86e48c0db1564a1d49548d3515ced5d604c408"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.9.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d4f69885afa5e6149d0cab3818491565cf41446d"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.4.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "e82c3c97b5b4ec111f3c1b55228cebc7510525a2"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.25"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "74911ad88921455c6afcad1eefa12bd7b1724631"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.80"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "38a92e40157100e796690421e34a11c107205c86"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "884477b9886a52a84378275737e2823a5c98e349"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "57f7cde02d7a53c9d1d28443b9f11ac5fbe7ebc9"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.3"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "e07a1b98ed72e3cdd02c6ceaab94b8a606faca40"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.2.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "fe9aea4ed3ec6afdfbeb5a4f39a2208909b162a6"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.5"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.GraphPlot]]
deps = ["ArnoldiMethod", "ColorTypes", "Colors", "Compose", "DelimitedFiles", "Graphs", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "5cd479730a0cb01f880eff119e9803c13f214cab"
uuid = "a2cc645c-3eea-5389-862e-a155d0052231"
version = "0.5.2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cf1d7dcb4bc32d7b4a5add4232db3750c27ecb4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.8.0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "678d136003ed5bceaab05cf64519e3f956ffa4ba"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.9.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "c3244ef42b7d4508c638339df1bdbf4353e144db"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.30"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "106b6aa272f294ba47e96bd3acbabdc0407b5c60"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyJSON]]
deps = ["JSON", "OrderedCollections", "PropertyDicts"]
git-tree-sha1 = "ce08411caa70e0c9e780f142f59debd89a971738"
uuid = "fc18253b-5e1b-504c-a4a2-9ece4944c004"
version = "0.2.2"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Librsvg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pango_jll", "Pkg", "gdk_pixbuf_jll"]
git-tree-sha1 = "ae0923dab7324e6bc980834f709c4cd83dd797ed"
uuid = "925c91fb-5dd6-59dd-8e8c-345e74382d89"
version = "2.54.5+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "46efcea75c890e5d820e670516dc156689851722"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.4"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "680e733c3a0a9cea9e935c8c2184aea6a63fa0b5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.21"

    [deps.LogExpFunctions.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    ChangesOfVariablesExt = "ChangesOfVariables"
    InverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "MiniQhull", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Setfield", "Showoff", "SignedDistanceFields", "SnoopPrecompile", "SparseArrays", "StableHashTraits", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "274fa9c60a10b98ab8521886eb4fe22d257dca65"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.19.2"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "2c3fc86d52dfbada1a2e5e150e50f06c30ef149c"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.6.2"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test", "UnicodeFun"]
git-tree-sha1 = "f04120d9adf4f49be242db0b905bea0be32198d1"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.4"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.MiniQhull]]
deps = ["QhullMiniWrapper_jll"]
git-tree-sha1 = "9dc837d180ee49eeb7c8b77bb1c860452634b0d1"
uuid = "978d7f02-9e05-4691-894f-ae31a51d76ca"
version = "0.4.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MolecularGraph]]
deps = ["DelimitedFiles", "JSON", "LinearAlgebra", "Printf", "Requires", "Statistics", "Unmarshal", "YAML", "coordgenlibs_jll", "libinchi_jll"]
git-tree-sha1 = "d59b9b59f8f3750f8110b2cd1d0c8ec408aab3fe"
uuid = "6c89ec66-9cd8-5372-9f91-fabc50dd27fd"
version = "0.12.0"

[[deps.MolecularGraphKernels]]
deps = ["Aqua", "Cairo", "Colors", "Compose", "Distributed", "GraphPlot", "Graphs", "JLD2", "MetaGraphs", "MolecularGraph", "PeriodicTable", "PrecompileSignatures", "ProgressMeter", "RDKitMinimalLib", "SharedArrays"]
git-tree-sha1 = "5b2628b172c03234e6515723a1a23986530cbb37"
uuid = "bf3818bd-b6bb-4954-8baa-32c32282e633"
version = "0.8.5"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Nullables]]
git-tree-sha1 = "8f87854cc8f3685a60689d8edecaa29d2251979b"
uuid = "4d1e1d77-625e-5b40-9113-a560ec7a8ecd"
version = "1.0.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "84a314e3926ba9ec66ac097e3635e270986b0f10"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.9+0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "18f84637e00b72ba6769034a4b50d79ee40c84a9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.5"

[[deps.PeriodicTable]]
deps = ["Base64", "Test", "Unitful"]
git-tree-sha1 = "5ed1e2691eb13b6e955aff1b7eec0b2401df208c"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.1.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileSignatures]]
git-tree-sha1 = "18ef344185f25ee9d51d80e179f8dad33dc48eb1"
uuid = "91cefc8d-f054-46dc-8f8c-26e11d7c5411"
version = "3.0.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.PropertyDicts]]
git-tree-sha1 = "8cf3b5cea994cfa9f238e19c3946a39cf051896c"
uuid = "f8a19df8-e894-5f55-a973-672c1158cbca"
version = "0.1.2"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "43d304ac6f0354755f1d60730ece8c499980f7ba"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QhullMiniWrapper_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Qhull_jll"]
git-tree-sha1 = "607cf73c03f8a9f83b36db0b86a3a9c14179621f"
uuid = "460c41e3-6112-5d7f-b78c-b6823adb3f2d"
version = "1.0.0+1"

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "238dd7e2cc577281976b9681702174850f8d4cbc"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "8.0.1001+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[deps.RDKitMinimalLib]]
deps = ["JSON", "RDKit_jll"]
git-tree-sha1 = "56837668e23c773b2537aceae7f3588ad4227077"
uuid = "44044271-7623-48dc-8250-42433c44e4b7"
version = "1.2.0"

[[deps.RDKit_jll]]
deps = ["Artifacts", "FreeType2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll", "boost_jll"]
git-tree-sha1 = "d8653fafde3ff0f0dde0cc4bb0e4d4820946fd33"
uuid = "03d1d220-30e6-562a-9e1a-3062d7b56d75"
version = "2022.9.4+0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rsvg]]
deps = ["Cairo", "Glib_jll", "Librsvg_jll"]
git-tree-sha1 = "3d3dc66eb46568fb3a5259034bfc752a0eb0c686"
uuid = "c4c386cf-5103-5370-be45-f3a111cca3b8"
version = "1.0.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "8b20084a97b004588125caebf418d8cab9e393d1"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.4"

[[deps.ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "2436b15f376005e8790e318329560dcc67188e84"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.3"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "c02bd3c9c3fc8463d3591a62a378f90d2d8ab0f3"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.17"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StableHashTraits]]
deps = ["CRC32c", "Compat", "Dates", "SHA", "Tables", "TupleTools", "UUIDs"]
git-tree-sha1 = "0b8b801b8f03a329a4e86b44c5e8a7d7f4fe10a3"
uuid = "c5dd0088-6c3f-4803-b00e-f31a60c170fa"
version = "0.3.1"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "129703d62117c374c4f2db6d13a027741c46eafd"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.13"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "a5e15f27abd2692ccb61a99e0854dfb7d48017db"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.33"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "33c0da881af3248dafefb939a21694b97cfece76"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.6"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "b03a3b745aa49b566f128977a7dd1be8711c5e71"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.14"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "7e6b0e3e571be0b4dd4d2a9a3a83b65c04351ccc"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.3"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d3f95a76c89777990d3d968ded5ecf12f9a0ad72"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.3"

[[deps.Unmarshal]]
deps = ["JSON", "LazyJSON", "Missings", "Nullables", "Requires"]
git-tree-sha1 = "ee46863309f8f942249e1df1b74ba3088ff0f151"
uuid = "cbff2730-442d-58d7-89d1-8e530c41eb02"
version = "0.4.4"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.YAML]]
deps = ["Base64", "Dates", "Printf", "StringEncodings"]
git-tree-sha1 = "dbc7f1c0012a69486af79c8bcdb31be820670ba2"
uuid = "ddb6d928-2868-570f-bddf-ab3f9cf99eb6"
version = "0.4.8"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

[[deps.boost_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7a89efe0137720ca82f99e8daa526d23120d0d37"
uuid = "28df3c45-c428-5900-9ff8-a3135698ca75"
version = "1.76.0+1"

[[deps.coordgenlibs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8a0fdb746dfc75758d0abea3196f5edfcbbebd79"
uuid = "f6050b86-aaaf-512f-8549-0afff1b4d57f"
version = "3.0.1+0"

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "e9190f9fb03f9c3b15b9fb0c380b0d57a3c8ea39"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.8+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinchi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "034ee07d3b387a4ca1a153a43a0c46549b6749ba"
uuid = "172afb32-8f1c-513b-968f-184fcd77af72"
version = "1.5.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╠═e4d22a48-8b16-4e7a-a776-fd12b6023123
# ╠═d5a773d7-3718-4cd1-be2a-68ad929bf051
# ╟─27ae40bf-6631-4737-9a3c-42a3de65964c
# ╟─e9799791-107d-4693-ae34-767a56d302c0
# ╟─27791708-984b-4ac1-bc70-fdfd97fe708e
# ╠═e1c1ead9-bebc-462e-9209-acb2aed030c5
# ╟─2711feb5-a6e5-4172-959d-60be301bdb80
# ╠═8926fafe-c057-43e0-bd8b-849e9af45ba7
# ╟─d5f1e207-0d57-476a-af6f-aa960fc55bd0
# ╠═39974a37-f5f1-48dc-b3e9-82c80583a570
# ╠═1cea56f6-4cf6-4005-a696-72ee77fadefb
# ╠═3d7a3459-afcd-45ba-a1e6-547d09226019
# ╠═d0bdc461-c415-4271-87be-294688883c8a
# ╠═439855bd-b35b-4f89-9a26-9ed670d48b16
# ╟─4cefde35-c896-430f-93bc-0b759c1f955d
# ╠═4e8566ac-bdd2-4842-8f72-28f2ea4c625f
# ╟─5c734e51-2ecd-4a38-b574-381fa88d52c9
# ╠═d539b75a-c1e6-4c71-933d-443fc2133330
# ╟─1df38111-f1a0-43c9-a2d5-6815100536c7
# ╠═f10eb262-c858-4269-80eb-d7344fd3e0c4
# ╟─fbe99ab0-c9af-4a8d-bbb7-708cbf674016
# ╠═51c34593-b29e-4e9e-96b4-bf2105ecae13
# ╟─fb9db5bd-9b94-4784-b91f-baae997d18b0
# ╟─4e0695f8-eff9-4709-9d74-841e97c2f5ad
# ╠═5c104d70-8bad-4fd7-a5db-39198e23efe1
# ╠═a76159c9-c4d0-4e3b-b25d-58fc2904edfa
# ╠═411255f2-2b8b-4a9b-9f4b-d3320541c702
# ╠═6e9a4254-6e55-4527-9db5-b9049d320428
# ╠═b57ca630-1bf8-4275-a583-09e257a670a2
# ╠═0fb36c69-0c34-4c03-93b5-3ad1b487559b
# ╠═491b2929-1c6b-41c4-b9c4-b155e4737d65
# ╟─8dee36a8-82ab-41b5-9ec5-7256efdfa5f5
# ╠═b2ececfd-69fc-41e6-a169-dc4ff1245373
# ╟─8cff11cf-60fa-4ce5-ad08-82075b00c308
# ╠═cd0f7408-5883-498f-b040-f68fbbdcd4bc
# ╟─31d7054e-ecab-4553-8588-30cbd4255b15
# ╠═3d67fd30-4c95-4f46-abab-daee2b155a08
# ╟─5aa3c3ea-e862-440b-92ed-7a0415f6ec5e
# ╠═93703e72-8718-44f1-8998-f38799efccce
# ╟─f2569c11-afe0-429a-a479-06a674936128
# ╠═cf866228-5674-481e-b9df-2bad986bc46c
# ╟─270b67f4-94d4-4008-957c-afa8084f5082
# ╠═c6b10ccd-2fa9-4fbb-aa59-8ad32fadf2b6
# ╟─24d34dc0-9b78-41dc-add0-8772a970b31c
# ╠═6ee51a94-8449-4fe7-896b-13507d9b3f8b
# ╟─308e3f12-00c2-4117-b4e7-8b469344189c
# ╟─e94ab1f3-4394-40b6-8223-10a3b7707606
# ╠═380a90ae-a0ae-445e-83a1-40d7ed85fafb
# ╟─309f21a9-47c6-469f-9ca9-62a7b5b9dfba
# ╠═1a7d0861-ec0d-4e8c-890e-48fe69c612c5
# ╟─9db8dbda-59cd-4e2a-ad8a-bf2978bedaba
# ╠═a0f25728-d129-405d-b715-db85f269a7d6
# ╟─bc402deb-5889-4a0d-b30c-d0e9295cb66e
# ╠═9b4e8cf6-8fe3-4022-870a-09f0d7933be5
# ╟─c976895a-f99e-478e-a44b-298fc8c841f2
# ╟─65986bc4-9f99-4e6f-af3e-a8989aa0f75a
# ╠═b1ac9d4f-dd01-4494-a9d8-a5d212ab366a
# ╠═9cea4488-3e06-46b9-b78b-2c971ddd2a24
# ╠═d51c2ae4-db6d-4d91-9f22-3a82deb74a6d
# ╠═83d308e1-f098-4d32-8e94-10d1db5a2eaa
# ╠═6e015806-c499-4439-bcbd-db57bca1d5a6
# ╠═8c95e70f-fbf2-4da4-a23a-30984ce03c9b
# ╠═796179c4-851f-485e-ba11-02c57bcd6b3c
# ╠═c384c30e-0b7e-4759-92ff-09b00f798613
# ╠═1840b5b4-dbcb-4ba2-a3ad-d97116619689
# ╟─bb2ed649-e2bb-4f27-a16f-a071fcb498f4
# ╠═1462bed5-ce9f-4969-bd99-f91fde1ba9e9
# ╟─c7aaa8a8-8b7a-4d6e-b1d8-73a6ef7ca559
# ╠═c76d435d-5e96-46a1-97ab-91e6d88f23cf
# ╟─05a38bf7-ea97-4d68-9381-db7c9488d29d
# ╠═0ac3ff6b-8123-4b40-939d-2e6a292997d4
# ╟─8714f0f2-8480-40cb-93c0-7c187739ec6e
# ╟─812a9f11-b2e0-4fa6-ac23-f77b7aee055a
# ╠═fd343574-2637-480e-9396-e3d1616ae8a9
# ╠═cc941a57-7ce9-4007-8a82-fa16744a4b45
# ╠═c024aa88-2850-436e-a830-ced1bd36807f
# ╟─8a787b8b-31e0-4928-bdba-a1afb61a1411
# ╠═6cba5e1a-31d5-4204-a5f1-78bd8e90dd18
# ╠═64f71860-501c-4db0-b498-bfb2477dc912
# ╠═61ff9250-07d9-459f-8f98-3572199529c4
# ╠═e45054ad-d11c-4eef-8623-504096934f53
# ╠═fbfe1bb0-80d1-4d4b-90f9-af60ab512ae4
# ╟─cb5a017e-6038-48fc-9ec0-49f0eee745d4
# ╠═9d4ff5db-7dc4-48e3-af8f-dc8e42914c78
# ╟─f5a9cde7-ef6c-4b81-9656-78727d9dd0ba
# ╠═64cfae63-0110-44dc-9335-c90d7d3a4370
# ╠═389b64ed-71b7-482b-a7c4-9ae392642a83
# ╟─9654fe84-8bbb-4937-8d7a-b9f0b7c1949b
# ╠═478b57a5-b74d-4142-bd55-6f73b9c076f1
# ╟─807c257c-24fe-4c4b-9965-03a8a58ff271
# ╟─3855a083-851a-41ec-9e42-1864c5e90b76
# ╠═f8294d1c-7a68-4196-ac7d-f78b00567d77
# ╟─964f1749-67e4-4f17-b6f5-2efc5017d4d3
# ╠═2f5e7378-b8cc-44ba-8eb8-c1ab85e92f79
# ╠═7cbe8e41-2571-4ea7-a359-6d2dbb87030a
# ╠═b59bd9a3-cc90-4bf9-ab42-6ea424df8f2d
# ╟─4b6dfaac-6d81-4343-8b5c-c40dd6e39129
# ╟─d20ffd27-335f-4e16-9f50-b48552bd6b22
# ╠═1eafe46f-1bfb-4cb1-b4bd-ef6b308f11bc
# ╠═95b5c3a3-1948-4ac9-a665-8103cf0afd66
# ╟─f8a2177a-efb5-43a3-9649-ef030429506d
# ╠═054fd209-287f-4e77-bbbc-7dd32e5a08aa
# ╟─8ec91912-31b8-40dc-ac6c-f9026bb5ccb6
# ╠═924dbe68-af24-4e7a-b285-84b4f22fbbb0
# ╠═26895aa5-d6e0-48e2-8811-bb7378d3e74d
# ╠═b5647646-b043-42e3-b104-5c1537c2970d
# ╠═81e3d848-3808-4af2-874a-8f38f1a79545
# ╠═ba788b21-ba0d-4a4a-9463-8778124240c7
# ╟─813d9850-f19a-49f0-bced-c71c50ba5b6c
# ╠═39c91036-b8a5-4082-97ee-1093e3fa4b28
# ╟─cc536782-4512-4786-b6af-ee6ce269931e
# ╠═2c3f3a16-98bf-4ed4-949a-9c46de02b3d2
# ╠═17649e4a-657d-468d-a393-fba0f4d881d4
# ╟─72cbc15d-59ee-49c7-a9d4-d5e3cdea9cd2
# ╠═4c1d1736-093d-4716-8317-6bd2072e6a82
# ╟─f05cfa94-0807-4305-8159-4836342bb161
# ╟─9185f35e-c3b8-4639-8b87-d6f60bcf9fe2
# ╠═30318671-b64f-4675-9954-3ce6953cf990
# ╟─074d3e61-3c3b-4b20-9d85-7f270b354c4d
# ╠═e4a0812b-7747-4c17-a57a-6d7247106fdf
# ╠═b6870321-407c-46d8-8c83-d8a3ee187cfe
# ╠═ce3a40ba-c942-4d92-8903-7f1906aab08d
# ╠═16ca287c-788e-4d87-8f16-1db41dda872f
# ╠═60a734a8-53ba-4908-b6c6-bccc9c93a938
# ╠═33026be4-fa61-4202-a3cf-d238273192a0
# ╠═83d9bc3b-cdeb-4b84-a8d3-c1cca85affea
# ╟─13301f55-094f-4f59-b0ed-faac030d090b
# ╟─e46ce0f0-76bc-4cac-83fd-3573ed62fd98
# ╠═d0832aad-4242-4b66-aae5-a8b5f09a46a0
# ╠═264f1fe3-62e1-4248-8129-d4d3c14af8ff
# ╠═1a5c3d6c-a941-4511-80c5-b1c67dcf8bb4
# ╟─be970511-03a3-4844-84b6-141e375ca99f
# ╟─356fc6a8-3363-48da-a802-494a090b120a
# ╠═cb00cedc-577b-44d1-aa78-3ba17083bdc0
# ╠═dbe3b4e3-1333-4f78-b22f-9a434607b9fc
# ╠═a4115745-8ab0-4231-9372-c130b3f8a4c1
# ╠═52282b6e-19de-4a2d-9258-be317a0b4d71
# ╠═eb3afc91-4207-44a1-a97f-8306ad5d6d75
# ╟─d92fc1dc-ae87-49d9-94e9-872f8c02f581
# ╠═a8dea5db-e306-47df-a2c7-a1da39e25b9b
# ╠═24e4487b-8117-43b1-bd09-feb2f38fe1bb
# ╟─1eb5ca97-bb40-4553-8c21-e3315230ecfc
# ╠═a3098e7b-c85d-40ac-a6f1-3f1c2f9f85da
# ╠═6467582e-837d-4c73-a168-054fe6e96fc8
# ╟─9e8a1722-1bee-42f7-a426-2a188e53582a
# ╠═e547a268-6188-471b-ab9e-5e5fc8433e1d
# ╠═a0be8244-930b-40f4-a1d5-16cdaa8161d0
# ╠═3284de1e-a1cb-4ed9-90a4-c3bfa535b3a6
# ╟─f2fa0edc-4633-4781-b032-7e7257bd63eb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
