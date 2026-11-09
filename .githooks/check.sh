#!/bin/sh
# Checks shared by the hooks: nothing personal goes into a commit. Lines that
# match are printed and the exit status is 1.
#
#   check.sh text <file>       a commit message (or - for stdin)
#   check.sh staged            the lines added in the staged changes, and gofmt
#   check.sh commits <rev...>  each commit's author, committer, message and
#                              added lines, for git rev-list <rev...>
#
# Besides the patterns here, each line is checked against the regular
# expressions in .git/info/private-patterns, one per line, if there is such a
# file: names and the like that must not be in the repository, even in a hook.

# Email addresses, except the no-reply ones and example domains.
email='[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}'
okemail='(4\+hdhrgit@noreply\.localhost|([0-9]+\+)?hdhrgit@users\.noreply\.github\.com|@[A-Za-z0-9.-]*\.(example|invalid|test))\b'
# Private addresses, except the examples the tests and docs use.
lan='\b(10\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])|192\.168)\.[0-9]{1,3}\.[0-9]{1,3}\b'
oklan='\b(10\.0\.0|10\.1\.1|10\.2\.0|10\.9\.9|192\.168\.1|192\.168\.5)\.[0-9]{1,3}\b'
# What a tool adds to say who asked for a change.
tag='Requested by|claude-projects-attribution|Reviewed-on'
# Authors and committers allowed.
okauthor='^(4\+hdhrgit@noreply\.localhost|([0-9]+\+)?hdhrgit@users\.noreply\.github\.com)$'

private="$(git rev-parse --git-common-dir)/info/private-patterns"

# scan reads lines and prints those that hold something personal. Only lines
# that may be are looked at one by one.
scan() {
	found=1
	tmp=$(mktemp)
	cat >"$tmp"
	{
		grep -E -e "$tag" -e "$email" -e "$lan" "$tmp"
		[ -s "$private" ] && grep -iE -f "$private" "$tmp"
	} | sort -u >"$tmp.hits"
	while IFS= read -r line; do
		bad=
		if printf '%s\n' "$line" | grep -Eq "$tag"; then
			bad="attribution"
		elif printf '%s\n' "$line" | grep -Eo "$email" | grep -Evq "$okemail"; then
			bad="email address"
		elif printf '%s\n' "$line" | grep -Eo "$lan" | grep -Evq "$oklan"; then
			bad="LAN address"
		elif [ -s "$private" ] && printf '%s\n' "$line" | grep -iEq -f "$private"; then
			bad="private pattern"
		fi
		if [ -n "$bad" ]; then
			printf '  %s: %s\n' "$bad" "$line"
			found=0
		fi
	done <"$tmp.hits"
	rm -f "$tmp" "$tmp.hits"
	[ "$found" = 1 ]
}

# added prints the lines a diff adds (the hooks themselves are left out by
# the pathspecs below: their patterns would match).
added() {
	grep '^+' | grep -v '^+++ '
}

ok=0
case "$1" in
text)
	file=${2:--}
	grep -v '^#' "$file" | scan || ok=1
	;;
staged)
	git diff --cached -U0 --no-color --no-ext-diff -- . ':(exclude).githooks' | added | scan || ok=1
	if command -v gofmt >/dev/null 2>&1; then
		for f in $(git diff --cached --name-only --diff-filter=ACM -- '*.go'); do
			if [ -n "$(git show ":$f" | gofmt -l)" ]; then
				printf '  not gofmt-formatted: %s\n' "$f"
				ok=1
			fi
		done
	fi
	;;
commits)
	shift
	for c in $(git rev-list "$@"); do
		out=$(
			for e in $(git log -1 --format='%ae %ce' "$c"); do
				printf '%s\n' "$e" | grep -Eq "$okauthor" || printf '  author or committer: %s\n' "$e"
			done
			git log -1 --format=%B "$c" | scan
			git show --format= -U0 --no-color --no-ext-diff "$c" -- . ':(exclude).githooks' | added | scan
		)
		if [ -n "$out" ]; then
			printf '%s %s\n%s\n' "$(git rev-parse --short "$c")" "$(git log -1 --format=%s "$c")" "$out"
			ok=1
		fi
	done
	;;
*)
	echo "usage: check.sh text <file> | staged | commits <rev...>" >&2
	exit 2
	;;
esac
exit $ok
