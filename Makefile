.PHONY: publish publish-derive publish-matma

# Publish both crates to crates.io in dependency order.
# cargo waits for matma-derive to land in the index before matma is verified.
publish: publish-derive publish-matma

publish-derive:
	cargo publish -p matma-derive

publish-matma:
	cargo publish -p matma
