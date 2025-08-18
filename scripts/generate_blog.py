#!/usr/bin/env python3
"""
Auto-generate blogs for church website using AWS Bedrock and existing blog examples.

This script:
1. Fetches existing blogs from S3 for multi-shot examples
2. Uses AWS Bedrock to generate new blog content (100% AI generated)
3. Uploads new blog to S3 or saves locally for testing
4. Updates the index.json file
"""

import json
import boto3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
import os
from botocore.exceptions import ClientError
from botocore.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BlogPost:
    """Represents a blog post with all its metadata."""
    slug: str
    title: str
    date: str
    summary: str
    path: str
    tags: List[str]
    image: str
    content: Optional[str] = None

class BlogGenerator:
    """Handles blog generation using existing examples and AWS Bedrock."""
    
    def __init__(self, bucket_name: str = "bethel-air-blogs", local_mode: bool = False, local_output_only: bool = False):
        self.bucket_name = bucket_name
        self.local_mode = local_mode  # Full local mode (read and write locally)
        self.local_output_only = local_output_only  # Read from S3, write locally
        self.s3_client = boto3.client('s3') if not (local_mode and not local_output_only) else None
        
        # Configure Bedrock client with 10-minute timeout
        bedrock_config = Config(
            read_timeout=600,  # 10 minutes in seconds
            connect_timeout=60,  # 1 minute for connection
            retries={'max_attempts': 3}  # Retry up to 3 times
        )
        self.bedrock_client = boto3.client('bedrock-runtime', config=bedrock_config)
        self.existing_blogs: Dict[str, BlogPost] = {}
        
        # Create local directories if in local mode
        if self.local_mode:
            os.makedirs('local_blogs/posts', exist_ok=True)
            os.makedirs('local_blogs/images', exist_ok=True)
        
    def fetch_blog_index(self) -> Dict:
        """Fetch the blog index from S3 or local filesystem."""
        if self.local_mode and not self.local_output_only:
            # Full local mode - read from local
            index_path = 'local_blogs/index.json'
            try:
                if os.path.exists(index_path):
                    with open(index_path, 'r') as f:
                        index_data = json.load(f)
                    logger.info(f"Fetched local blog index with {len(index_data.get('posts', []))} posts")
                    return index_data
                else:
                    logger.info("No local index found, creating new one")
                    return {"updated": datetime.now(timezone.utc).isoformat(), "posts": []}
            except Exception as e:
                logger.error(f"Error fetching local blog index: {e}")
                return {"updated": datetime.now(timezone.utc).isoformat(), "posts": []}
        else:
            # Read from S3 (either normal mode or local_output_only mode)
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key='blogs/index.json'
                )
                index_data = json.loads(response['Body'].read().decode('utf-8'))
                logger.info(f"Fetched blog index with {len(index_data.get('posts', []))} posts")
                return index_data
            except Exception as e:
                logger.error(f"Error fetching blog index: {e}")
                return {"updated": datetime.now(timezone.utc).isoformat(), "posts": []}
    
    def fetch_blog_content(self, blog_path: str) -> Optional[str]:
        """Fetch the markdown content of a specific blog post."""
        if self.local_mode and not self.local_output_only:
            # Full local mode - read from local
            local_path = f"local_blogs/{blog_path.replace('blogs/', '')}"
            try:
                if os.path.exists(local_path):
                    with open(local_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.info(f"Fetched local content for {local_path}")
                    return content
                else:
                    logger.warning(f"Local file not found: {local_path}")
                    return None
            except Exception as e:
                logger.error(f"Error fetching local blog content from {local_path}: {e}")
                return None
        else:
            # Read from S3 (either normal mode or local_output_only mode)
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=blog_path
                )
                content = response['Body'].read().decode('utf-8')
                logger.info(f"Fetched content for {blog_path}")
                return content
            except Exception as e:
                logger.error(f"Error fetching blog content from {blog_path}: {e}")
                return None
    
    def load_existing_blogs(self, max_blogs: int = 10) -> None:
        """Load existing blogs for use as multi-shot examples."""
        index = self.fetch_blog_index()
        posts = index.get('posts', [])
        
        # Get the most recent posts (up to max_blogs)
        recent_posts = sorted(posts, key=lambda x: x['date'], reverse=True)[:max_blogs]
        
        logger.info(f"Loading {len(recent_posts)} existing blogs for examples")
        
        for post_data in recent_posts:
            # Create BlogPost object
            blog_post = BlogPost(
                slug=post_data['slug'],
                title=post_data['title'],
                date=post_data['date'],
                summary=post_data['summary'],
                path=post_data['path'],
                tags=post_data['tags'],
                image=post_data['image']
            )
            
            # Fetch the actual content
            content = self.fetch_blog_content(post_data['path'])
            if content:
                blog_post.content = content
                self.existing_blogs[blog_post.slug] = blog_post
            else:
                logger.warning(f"Could not fetch content for {post_data['slug']}")
        
        logger.info(f"Successfully loaded {len(self.existing_blogs)} blogs with content")
    
    def create_multishot_examples(self) -> List[Dict[str, str]]:
        """Create multi-shot examples from existing blogs for the LLM."""
        examples = []
        
        for slug, blog in self.existing_blogs.items():
            # Create a simplified version for the example
            example = {
                "title": blog.title,
                "summary": blog.summary,
                "tags": ", ".join(blog.tags),
                "content_preview": blog.content[:500] + "..." if blog.content and len(blog.content) > 500 else blog.content or ""
            }
            examples.append(example)
        
        return examples
    
    def build_full_prompt(self, base_prompt: str, examples: List[Dict[str, str]]) -> str:
        """Combine the base prompt with dynamic examples from S3."""
        # Build the examples text
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += f"\n### Example {i}:\n"
            examples_text += f"**Title:** {example['title']}\n"
            examples_text += f"**Summary:** {example['summary']}\n"
            examples_text += f"**Tags:** {example['tags']}\n"
            examples_text += f"**Content Preview:** {example['content_preview']}\n"
        
        # Replace the placeholder with actual examples
        full_prompt = base_prompt.replace("{EXAMPLES_PLACEHOLDER}", examples_text)
        
        return full_prompt
    
    def generate_blog_with_bedrock(self, examples: List[Dict[str, str]], base_prompt: str = "") -> Dict[str, str]:
        """Generate blog content using AWS Bedrock with dynamic examples."""
        logger.info("Generating fully AI-generated blog content")
        
        # Build the complete prompt with examples
        full_prompt = self.build_full_prompt(base_prompt, examples)
        logger.info(f"Generated prompt with {len(examples)} examples")
        
        # Set the model ID for Claude Opus 4.1 - use inference profile
        model_id = "us.anthropic.claude-opus-4-1-20250805-v1:0"
        
        # Prepare the conversation for Bedrock
        conversation = [
            {
                "role": "user",
                "content": [{"text": full_prompt}],
            }
        ]
        
        try:
            logger.info(f"Calling Bedrock with model: {model_id}")
            
            # Call Bedrock with Claude Opus 4.1
            response = self.bedrock_client.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig={
                    "maxTokens": 4000,  # Increased for longer blog posts
                    "temperature": 0.7,  # Balanced creativity
                    "topP": 0.9
                }
            )
            
            # Extract the response text
            response_text = response["output"]["message"]["content"][0]["text"]
            logger.info("Successfully received response from Bedrock")
            
            # Parse the JSON response
            try:
                # Handle case where Claude wraps JSON in markdown code blocks
                response_text_clean = response_text.strip()
                if response_text_clean.startswith('```json'):
                    # Extract JSON from markdown code block
                    start_idx = response_text_clean.find('```json') + 7
                    end_idx = response_text_clean.rfind('```')
                    if end_idx > start_idx:
                        response_text_clean = response_text_clean[start_idx:end_idx].strip()
                elif response_text_clean.startswith('```'):
                    # Handle generic code block
                    start_idx = response_text_clean.find('```') + 3
                    end_idx = response_text_clean.rfind('```')
                    if end_idx > start_idx:
                        response_text_clean = response_text_clean[start_idx:end_idx].strip()
                
                blog_data = json.loads(response_text_clean)
                
                # Validate required fields
                required_fields = ["title", "content", "summary", "tags"]
                for field in required_fields:
                    if field not in blog_data:
                        raise ValueError(f"Missing required field '{field}' in Bedrock response")
                
                # Ensure tags is a list
                if isinstance(blog_data["tags"], str):
                    # If tags is a string, try to split it
                    blog_data["tags"] = [tag.strip() for tag in blog_data["tags"].split(",")]
                elif not isinstance(blog_data["tags"], list):
                    raise ValueError(f"Tags field must be a list or comma-separated string, got: {type(blog_data['tags'])}")
                
                logger.info(f"Successfully parsed blog: '{blog_data['title']}'")
                return blog_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from Bedrock: {e}")
                logger.error(f"Raw response: {response_text[:500]}...")
                raise ValueError(f"Bedrock returned invalid JSON: {e}") from e
                
            except ValueError as e:
                logger.error(f"Invalid response format from Bedrock: {e}")
                raise ValueError(f"Bedrock response validation failed: {e}") from e
        
        except ClientError as e:
            logger.error(f"AWS Bedrock ClientError: {e}")
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"Error Code: {error_code}, Message: {error_message}")
            raise RuntimeError(f"Bedrock API error [{error_code}]: {error_message}") from e
            
        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock: {e}")
            raise RuntimeError(f"Unexpected error during Bedrock call: {e}") from e
    
    def generate_slug(self, title: str) -> str:
        """Generate a URL-friendly slug from a title."""
        # Convert to lowercase and replace spaces with hyphens
        slug = re.sub(r'[^\w\s-]', '', title.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')
    
    def save_blog_content(self, slug: str, content: str, metadata: Dict) -> str:
        """Save blog content either to S3 or locally based on mode."""
        blog_path = f"blogs/posts/{slug}.md"
        
        # Create frontmatter header
        frontmatter = f"""---
title: "{metadata['title']}"
date: {metadata['date']}
description: "{metadata.get('summary', '')}"
---

"""
        
        # Remove title from content if it exists as first line
        content_lines = content.split('\n')
        if content_lines and content_lines[0].startswith('# '):
            # Remove the title line and any following empty lines
            while content_lines and (content_lines[0].startswith('# ') or content_lines[0].strip() == ''):
                content_lines.pop(0)
            content = '\n'.join(content_lines)
        
        # Combine frontmatter with content
        full_content = frontmatter + content
        
        if self.local_mode or self.local_output_only:
            local_path = f"local_blogs/posts/{slug}.md"
            try:
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(full_content)
                logger.info(f"Saved blog locally to {local_path}")
                return blog_path
            except Exception as e:
                logger.error(f"Error saving blog locally: {e}")
                raise
        else:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=blog_path,
                    Body=full_content.encode('utf-8'),
                    ContentType='text/markdown'
                )
                logger.info(f"Uploaded blog to {blog_path}")
                return blog_path
            except Exception as e:
                logger.error(f"Error uploading blog to S3: {e}")
                raise
    
    def update_blog_index(self, new_post: BlogPost) -> None:
        """Update the blog index with the new post."""
        index = self.fetch_blog_index()
        
        # Add new post to the beginning of the list
        new_post_data = {
            "slug": new_post.slug,
            "title": new_post.title,
            "date": new_post.date,
            "summary": new_post.summary,
            "path": new_post.path,
            "tags": new_post.tags,
            "image": new_post.image
        }
        
        index['posts'].insert(0, new_post_data)
        index['updated'] = datetime.now(timezone.utc).isoformat()
        
        if self.local_mode or self.local_output_only:
            local_path = 'local_blogs/index.json'
            try:
                with open(local_path, 'w', encoding='utf-8') as f:
                    json.dump(index, f, indent=2)
                logger.info(f"Updated local blog index at {local_path}")
            except Exception as e:
                logger.error(f"Error updating local blog index: {e}")
                raise
        else:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key='blogs/index.json',
                    Body=json.dumps(index, indent=2).encode('utf-8'),
                    ContentType='application/json'
                )
                logger.info("Updated blog index successfully")
            except Exception as e:
                logger.error(f"Error updating blog index: {e}")
                raise
    
    def run_generation(self, custom_prompt: Optional[str] = None) -> BlogPost:
        """Main method to generate a new blog post (100% AI generated)."""
        logger.info("Starting fully AI-generated blog creation")
        
        # Load existing blogs for examples
        self.load_existing_blogs()
        
        # Create multi-shot examples
        examples = self.create_multishot_examples()
        logger.info(f"Created {len(examples)} multi-shot examples")
        
        # Use custom prompt or empty string if none provided
        base_prompt = custom_prompt or ""
        
        # Generate content (100% AI generated)
        generated_content = self.generate_blog_with_bedrock(examples, base_prompt)
        
        # Create BlogPost object
        slug = self.generate_slug(generated_content['title'])
        today = datetime.now().strftime('%Y-%m-%d')
        
        new_blog = BlogPost(
            slug=slug,
            title=generated_content['title'],
            date=today,
            summary=generated_content['summary'],
            path=f"blogs/posts/{slug}.md",
            tags=generated_content['tags'],
            image=f"blogs/images/{slug}-hero.v1.jpg",  # You'll need to handle image generation separately
            content=generated_content['content']
        )
        
        # Save content (either to S3 or locally)
        blog_path = self.save_blog_content(slug, new_blog.content, {
            'title': new_blog.title,
            'date': new_blog.date,
            'summary': new_blog.summary,
            'tags': new_blog.tags
        })
        
        # Update index
        self.update_blog_index(new_blog)
        
        storage_location = "locally" if self.local_mode else "to S3"
        logger.info(f"Successfully generated and saved blog {storage_location}: {new_blog.title}")
        return new_blog

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate church blog posts using AI (100% AI generated)')
    parser.add_argument('--bucket', default='bethel-air-blogs', help='S3 bucket name')
    parser.add_argument('--prompt', help='Custom prompt file path (default: scripts/prompts/generate_blog.txt)')
    parser.add_argument('--local', action='store_true', help='Save files locally instead of uploading to S3 (for testing)')
    parser.add_argument('--local-output-only', action='store_true', help='Read examples from S3 but save output locally')
    
    args = parser.parse_args()
    
    # Initialize generator
    if args.local_output_only:
        generator = BlogGenerator(bucket_name=args.bucket, local_output_only=True)
    else:
        generator = BlogGenerator(bucket_name=args.bucket, local_mode=args.local)
    
    # Load prompt - use default or custom
    custom_prompt = None
    prompt_path = args.prompt if args.prompt else 'scripts/prompts/generate_blog.txt'
    
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            custom_prompt = f.read()
        logger.info(f"Loaded prompt from: {prompt_path}")
    else:
        logger.warning(f"Prompt file not found: {prompt_path}")
        if not args.prompt:  # Only warn if using default path
            logger.warning("Using default prompt - consider creating scripts/prompts/generate_blog.txt")
    
    try:
        # Generate the blog (no topic needed - 100% AI generated)
        new_blog = generator.run_generation(custom_prompt)
        
        storage_info = "saved locally in ./local_blogs/" if args.local else "uploaded to S3"
        print(f"✅ Successfully generated blog: '{new_blog.title}'")
        print(f"📝 Slug: {new_blog.slug}")
        print(f"🔗 Path: {new_blog.path}")
        print(f"💾 Storage: {storage_info}")
        
        if args.local_output_only:
            # Output blog details for GitHub Actions to capture
            print(f"BLOG_TITLE={new_blog.title}")
            print(f"BLOG_SUMMARY={new_blog.summary}")
            print(f"BLOG_CONTENT={new_blog.content}")
            print(f"BLOG_SLUG={new_blog.slug}")
            print(f"BLOG_TAGS={', '.join(new_blog.tags)}")
        
    except Exception as e:
        logger.error(f"Failed to generate blog: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())