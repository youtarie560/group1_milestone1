# frozen_string_literal: true

source "https://rubygems.org"

git_source(:github) { |repo_name| "https://github.com/#{repo_name}" }

# Core site generator
gem "jekyll", "~> 4.3.3"

# Fix Ruby 3.4 stdlib removal
gem "csv", require: true
gem "base64", require: true


# Plugins
gem "jekyll-gist"
gem "jekyll-sitemap"
gem "jekyll-seo-tag"
gem "jekyll-paginate"

# Required for Ruby 3.x
gem "webrick", "~> 1.7"