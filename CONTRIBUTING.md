# Contributing

Want to contribute? Great! You can do so through the standard GitHub pull
request model. For large contributions we do encourage you to file a ticket in
the GitHub issues tracking system prior to any code development to coordinate
with the pyLife development team early in the process. Coordinating up
front helps to avoid frustration later on.

Your contribution must be licensed under the Apache-2.0 license, the license
used by this project.

## Making commits

### Configure your git client

Please configure your identity in your git client appropriately. From the git
command line you can do that using
```
git config user.name <Your Name>
git config user.email <your-email@...>
```

## Add / retain copyright notices

Include a copyright notice and license in each new file to be contributed,
consistent with the style used by this project. If your contribution contains
code under the copyright of a third party, document its origin, license, and
copyright holders.

## Sign your work

This project tracks patch provenance and licensing using a modified Developer
Certificate of Origin (DCO; from [OSDL][DCO]) and Signed-off-by tags initially
developed by the Linux kernel project.

```

pyLife Developer's Certificate of Origin.  Version 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the "Apache License, Version 2.0"
    ("Apache-2.0"); or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.

(e) I am granting the contribution to this project under the terms of
    Apache-2.0.

    http://www.apache.org/licenses/LICENSE-2.0
```

With the sign-off in a commit message you certify that you authored the patch
or otherwise have the right to submit it under an open source license. The
procedure is simple: To certify above pyLife Developer's Certificate of
Origin 1.1 for your contribution just append a line

    Signed-off-by: Random J Developer <random@developer.example.org>

to every commit message using your real name or your pseudonym and a valid
email address.

If you have set your `user.name` and `user.email` git configs you can
automatically sign the commit by running the git-commit command with the `-s`
option.  There may be multiple sign-offs if more than one developer was
involved in authoring the contribution.

Another option to automatically add the `Signed-off-by:` is to once use the
command
```
git config core.hooksPath .githooks
```
in your pyLife working directory. This will then add the `Signed-off-by:` line
automatically.

For a more detailed description of this procedure, please see
[SubmittingPatches][] which was extracted from the Linux kernel project, and
which is stored in an external repository.

### Individual vs. Corporate Contributors

Often employers or academic institution have ownership over code that is
written in certain circumstances, so please do due diligence to ensure that
you have the right to submit the code.

If you are a developer who is authorized to contribute to pyLife on
behalf of your employer, then please use your corporate email address in the
Signed-off-by tag. Otherwise please use a personal email address.

## Maintain Copyright holder / Contributor list

Each contributor is responsible for identifying themselves in the
[NOTICE](NOTICE) file, the project's list of copyright holders and authors.
Please add the respective information corresponding to the Signed-off-by tag
as part of your first pull request.

If you are a developer who is authorized to contribute to pyLife on
behalf of your employer, then add your company / organization to the list of
copyright holders in the [NOTICE](NOTICE) file. As author of a corporate
contribution you can also add your name and corporate email address as in the
Signed-off-by tag.

If your contribution is covered by this project's DCO's clause "(c) The
contribution was provided directly to me by some other person who certified
(a) or (b) and I have not modified it", please add the appropriate copyright
holder(s) to the [NOTICE](NOTICE) file as part of your contribution.

[pytest]: https://pytest.org

[rrugamba]: https://medium.com/@rrugamba/3-laws-of-tdd-58b5ec46a998

[CGL]: https://www.git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_commit_guidelines

[git-commit]: https://chris.beams.io/posts/git-commit/

[DCO]: http://web.archive.org/web/20070306195036/http://osdlab.org/newsroom/press_releases/2004/2004_05_24_dco.html

[SubmittingPatches]: https://github.com/wking/signed-off-by/blob/7d71be37194df05c349157a2161c7534feaf86a4/Documentation/SubmittingPatches
