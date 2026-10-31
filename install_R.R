# Run this in an R console, not in the Python kernel.
packages <- c("IRkernel", "fixest", "arrow", "data.table", "jsonlite", "tidyselect")
missing <- packages[!vapply(packages,requireNamespace,logical(1),quietly=TRUE)]
if (length(missing)) install.packages(missing,repos="https://cloud.r-project.org")
IRkernel::installspec(user=TRUE,name="ir",displayname="R (SciSciNet2 fixest)")
print(sessionInfo())
