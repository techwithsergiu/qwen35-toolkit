import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'Qwen35 toolkit',
    description: 'Convert, strip, verify and publish Qwen3.5 models on your hardware',
    base: '/qwen35-toolkit/',

    head: [
      ['link', { rel: 'icon', href: '/qwen35-toolkit/favicon.ico' }],
    ],

    themeConfig: {
      nav: [
        { text: 'Quickstart', link: '/quickstart' },
        {
          text: 'Reference',
          items: [
            { text: 'Conversion pipeline', link: '/conversion-pipeline' },
            { text: 'Convert', link: '/convert' },
            { text: 'Strip', link: '/strip' },
            { text: 'Verify', link: '/verify' },
            { text: 'Upload', link: '/upload' },
            { text: 'GGUF', link: '/gguf' },
            { text: 'Tools', link: '/tools' },
          ],
        },
        { text: 'Published models', link: '/models' },
        { text: 'GitHub', link: 'https://github.com/techwithsergiu/qwen35-toolkit' },
      ],

      sidebar: [
        {
          text: 'Getting started',
          items: [
            { text: 'Overview', link: '/' },
            { text: 'Setup', link: '/setup' },
            { text: 'Quickstart', link: '/quickstart' },
            { text: 'Commands', link: '/commands' },
          ],
        },
        {
          text: 'Reference',
          items: [
            { text: 'Conversion pipeline', link: '/conversion-pipeline' },
            { text: 'Convert', link: '/convert' },
            { text: 'Strip', link: '/strip' },
            { text: 'Verify', link: '/verify' },
            { text: 'Upload', link: '/upload' },
            { text: 'GGUF', link: '/gguf' },
            { text: 'Tools', link: '/tools' },
          ],
        },
        {
          text: 'Misc',
          items: [
            { text: 'Published Models', link: '/models' },
            { text: 'Hardware', link: '/hardware' },
            { text: 'Third-party Licenses', link: '/THIRD_PARTY_LICENSES' },
          ],
        },
      ],

      socialLinks: [
        { icon: 'github', link: 'https://github.com/techwithsergiu/qwen35-toolkit' },
      ],

      footer: {
        message: 'Released under the Apache 2.0 License.',
        copyright: 'Part of the <a href="https://github.com/techwithsergiu/qwen-qlora-train">qwen-qlora-train</a> ecosystem.',
      },

      editLink: {
        pattern: 'https://github.com/techwithsergiu/qwen35-toolkit/edit/main/docs/:path',
        text: 'Edit this page on GitHub',
      },
    },

    mermaid: {
      theme: 'default',
    },
  })
)
